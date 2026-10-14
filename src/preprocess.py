"""pcap → 플로우마다 앞 30개 패킷의 (간격, 방향, 크기) 행렬.

CESNET-QUIC22의 PPI와 같은 모양 (N, 3, 30)으로 만든다.
- 행 순서: IPT(ms, 첫 패킷은 0) · DIR(+1 클라이언트→서버, −1 서버→클라이언트) · SIZE(L4 페이로드 바이트)
- 30개보다 짧은 플로우는 0으로 채운다.
- TCP는 CESNET과 같게 페이로드가 0인 패킷(순수 ACK 등)과 SEQ가 늘지 않는 패킷(재전송)을 뺀다.
  QUIC(UDP)은 모두 넣는다.
- 클라이언트 = 플로우의 첫 패킷을 보낸 쪽.
- 캡처 전에 열려 중간부터 잡힌 플로우(TCP는 SYN, QUIC은 long header로 시작하지 않음)는 뺀다.
- offset = 캡처 시작부터 플로우가 열릴 때까지 걸린 초. 캡처를 켤 때 몰리는 재연결을 거를 때 쓴다.

IP·포트는 플로우를 묶고 거르는 데만 쓴다. 모델 입력(ppi)에는 넣지 않는다.

    python -m src.preprocess --status   # 파일 이름 검사 + 라벨×날짜 파일 수 (tshark 불필요)
    python -m src.preprocess            # 직접 수집 pcap 전체 → processed/self_collected.npz
"""
from __future__ import annotations

import argparse
import io
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DIR, IPT, LABELS, N_PACKETS, PROCESSED_DIR, SELF_COLLECTED_DIR, SIZE, TASKS

TSHARK_FIELDS = [
    "frame.time_epoch",
    "ip.src", "ip.dst", "ipv6.src", "ipv6.dst",
    "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport",
    "tcp.len", "udp.length", "tcp.seq", "tcp.flags",
    "quic.header_form",  # 1 = long header (핸드셰이크), 0 = short header (연결 도중)
]
# ICMP 오류 메시지 안에 든 TCP·UDP 헤더는 뺀다
TSHARK_FILTER = "(tcp or udp) and not icmp and not icmpv6"

TCP_SYN = 0x02
TCP_ACK = 0x10

# 직접 수집 파일 이름: 라벨_날짜_순번.pcap (예: youtube_1013_1.pcap, instagram-reels_1013_1.pcap)
PCAP_NAME = re.compile(r"^(?P<label>[a-z0-9-]+)_(?P<date>\d{4})_(?P<seq>\d+)\.pcap(ng)?$")

PACKET_COLUMNS = ["time", "src", "dst", "sport", "dport", "proto", "payload", "seq", "flags", "quic_long"]

# 앱 분류 파일에서 중간부터 잡힌 연결이 443 패킷의 이 비율을 넘으면 경고한다
PARTIAL_WARN = 0.3


def parse_pcap_name(name: str) -> dict | None:
    """'instagram-reels_1013_1.pcap' → {'label': 'instagram-reels', 'date': '1013', 'session': 'instagram-reels_1013_1'}.
    규칙에 안 맞거나 라벨이 config.LABELS에 없으면 None."""
    m = PCAP_NAME.match(name)
    if m is None or m["label"] not in LABELS:
        return None
    return {"label": m["label"], "date": m["date"], "session": Path(name).stem}


def find_tshark() -> str:
    path = shutil.which("tshark") or r"C:\Program Files\Wireshark\tshark.exe"
    if not Path(path).exists():
        raise FileNotFoundError("tshark를 찾을 수 없다. Wireshark를 설치한다 (tshark 포함).")
    return path


def run_tshark(pcap: Path) -> str:
    """pcap → 패킷마다 한 줄인 CSV 문자열."""
    cmd = [find_tshark(), "-r", str(pcap), "-n", "-Y", TSHARK_FILTER, "-T", "fields",
           "-E", "header=y", "-E", "separator=,", "-E", "occurrence=f"]
    for field in TSHARK_FIELDS:
        cmd += ["-e", field]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=True).stdout


def packets_from_tshark(csv_text: str) -> pd.DataFrame:
    """tshark CSV → 패킷 표 (time, src, dst, sport, dport, proto, payload, seq, flags, quic_long)."""
    if not csv_text.strip():
        return pd.DataFrame(columns=PACKET_COLUMNS)
    raw = pd.read_csv(io.StringIO(csv_text), dtype=str)
    is_tcp = raw["tcp.srcport"].notna()
    return pd.DataFrame({
        "time": raw["frame.time_epoch"].astype(float),
        "src": raw["ip.src"].fillna(raw["ipv6.src"]),
        "dst": raw["ip.dst"].fillna(raw["ipv6.dst"]),
        "sport": raw["tcp.srcport"].fillna(raw["udp.srcport"]).astype(int),
        "dport": raw["tcp.dstport"].fillna(raw["udp.dstport"]).astype(int),
        "proto": np.where(is_tcp, "tcp", "udp"),
        # udp.length는 UDP 헤더 8바이트를 포함한다
        "payload": np.where(is_tcp, pd.to_numeric(raw["tcp.len"]), pd.to_numeric(raw["udp.length"]) - 8),
        "seq": pd.to_numeric(raw["tcp.seq"]),
        "flags": raw["tcp.flags"].map(lambda v: int(v, 0), na_action="ignore"),
        # tshark 버전에 따라 불리언을 1/0 또는 True/False로 쓴다
        "quic_long": raw["quic.header_form"].isin(["1", "True", "true"]),
    })


def flows_from_packets(
    pk: pd.DataFrame,
    server_ports: tuple[int, ...] | None = (443,),
    drop_partial: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """패킷 표 → (ppi (N, 3, 30) float32, 플로우 정보 표).

    server_ports: 이 포트로 접속한 플로우만 남긴다 (443 = TLS·QUIC). None이면 모두.
    drop_partial: 캡처 전에 열려 중간부터 잡힌 플로우를 뺀다. TCP는 SYN, UDP는 QUIC long header
        패킷(핸드셰이크)으로 시작하지 않으면 중간부터 잡힌 것이다. 첫 패킷이 없으면 클라이언트·방향을
        잘못 잡고, 앞 30개 패킷도 진짜 시작이 아니다. 인스타 화면 분류처럼 앱을 켜 둔 채 캡처를 켤 때 특히 많다.
    """
    if pk.empty:
        empty_meta = pd.DataFrame(columns=["proto", "server_port", "client_port", "start", "offset", "starts_with_handshake",
                                           "packets", "ppi_len"])
        return np.zeros((0, 3, N_PACKETS), dtype=np.float32), empty_meta
    pk = pk.sort_values("time", kind="stable").reset_index(drop=True)
    a = pk["src"] + "|" + pk["sport"].astype(str)
    b = pk["dst"] + "|" + pk["dport"].astype(str)
    key = pk["proto"].astype(str) + "|" + (a + "|" + b).where(a < b, b + "|" + a)
    pk["flow"] = pd.factorize(key)[0]  # 처음 나온 순서 = 시간 순서

    first = pk.groupby("flow").head(1).set_index("flow")
    client = first["src"] + "|" + first["sport"].astype(str)
    pk["dir"] = np.where(a.to_numpy() == client.reindex(pk["flow"]).to_numpy(), 1, -1)

    flags = first["flags"].fillna(0).astype(int)
    syn_start = ((flags & TCP_SYN) > 0) & ((flags & TCP_ACK) == 0)
    quic_start = first["quic_long"].fillna(False).astype(bool)
    meta = pd.DataFrame({
        "proto": first["proto"],
        "server_port": first["dport"],
        "client_port": first["sport"],  # 중간부터 잡힌 연결은 서버가 먼저 보내면 둘이 뒤바뀐다
        "start": first["time"],
        "offset": first["time"] - pk["time"].iloc[0],  # 캡처의 첫 패킷부터 플로우가 열릴 때까지 초
        "starts_with_handshake": syn_start.where(first["proto"] == "tcp", quic_start),
        "packets": pk.groupby("flow").size(),
    })
    keep = pd.Series(True, index=meta.index)
    if server_ports is not None:
        keep &= meta["server_port"].isin(server_ports)
    if drop_partial:
        keep &= meta["starts_with_handshake"]

    pk = pk[pk["flow"].isin(meta.index[keep])]
    is_tcp = pk["proto"] == "tcp"
    pk = pk[~is_tcp | (pk["payload"] > 0)]
    # TCP 재전송: 같은 방향에서 SEQ가 앞서 본 최대값보다 크지 않으면 뺀다
    prev_max = pk.groupby(["flow", "dir"])["seq"].cummax().groupby([pk["flow"], pk["dir"]]).shift()
    pk = pk[(pk["proto"] != "tcp") | prev_max.isna() | (pk["seq"] > prev_max)].copy()

    pk["pos"] = pk.groupby("flow").cumcount()
    pk["ipt"] = pk.groupby("flow")["time"].diff().fillna(0.0) * 1000.0
    pk = pk[pk["pos"] < N_PACKETS]

    ppi_len = pk.groupby("flow").size()
    meta = meta.loc[ppi_len.index].assign(ppi_len=ppi_len)
    row = pd.Series(np.arange(len(meta)), index=meta.index).loc[pk["flow"]].to_numpy()
    col = pk["pos"].to_numpy()

    ppi = np.zeros((len(meta), 3, N_PACKETS), dtype=np.float32)
    ppi[row, IPT, col] = pk["ipt"].to_numpy()
    ppi[row, DIR, col] = pk["dir"].to_numpy()
    ppi[row, SIZE, col] = pk["payload"].to_numpy()
    return ppi, meta.reset_index(drop=True)


def pcap_to_flows(pcap: Path, **kwargs) -> tuple[np.ndarray, pd.DataFrame]:
    """pcap 하나 → (ppi, 플로우 정보). 파일 이름이 규칙에 맞으면 label·date·session 열을 붙인다."""
    ppi, meta = flows_from_packets(packets_from_tshark(run_tshark(pcap)), **kwargs)
    label = parse_pcap_name(Path(pcap).name)
    if label is not None:
        meta = meta.assign(**label)
    return ppi, meta


def list_pcaps(folder: Path = SELF_COLLECTED_DIR) -> tuple[list[Path], list[Path]]:
    """(이름 규칙에 맞는 파일, 안 맞는 파일)."""
    files = sorted(p for p in Path(folder).iterdir() if p.suffix in (".pcap", ".pcapng"))
    valid = [p for p in files if parse_pcap_name(p.name)]
    return valid, [p for p in files if p not in valid]


def collection_status(folder: Path = SELF_COLLECTED_DIR) -> pd.DataFrame:
    """라벨 × 날짜 파일 수 표."""
    valid, _ = list_pcaps(folder)
    names = pd.DataFrame([parse_pcap_name(p.name) for p in valid], columns=["label", "date", "session"])
    return pd.crosstab(names["label"], names["date"], margins=True, margins_name="합계")


def build_self_collected(folder: Path = SELF_COLLECTED_DIR, out: Path = PROCESSED_DIR / "self_collected.npz") -> Path:
    """직접 수집 pcap 전체 → npz (ppi, label, date, session, proto, offset)."""
    valid, invalid = list_pcaps(folder)
    for p in invalid:
        print(f"건너뜀 (이름 규칙 위반): {p.name}")
    ppis, metas = [], []
    for p in valid:
        packets = packets_from_tshark(run_tshark(p))
        ppi, meta = flows_from_packets(packets)
        name = parse_pcap_name(p.name)
        # 중간부터 잡혀 뺀 연결이 트래픽 대부분이면, 앱 분류 파일은 캡처 전에 앱이 살아 있었던 것이다
        _, every = flows_from_packets(packets, server_ports=None, drop_partial=False)
        every = every[(every["server_port"] == 443) | (every["client_port"] == 443)]
        partial = every[~every["starts_with_handshake"]]
        share = partial["packets"].sum() / max(every["packets"].sum(), 1)
        line = f"{p.name}: 플로우 {len(meta)}개 (중간부터 잡혀 뺀 연결 {len(partial)}개 = 443 패킷의 {share:.0%})"
        if name["label"] in TASKS["app"] and share > PARTIAL_WARN:
            line += " ← 앱을 강제 종료하지 않고 캡처를 켠 것 같다"
        print(line)
        ppis.append(ppi)
        metas.append(meta.assign(**name))
    if not metas:
        raise SystemExit(f"{folder}에 처리할 pcap이 없다.")
    meta = pd.concat(metas, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        ppi=np.concatenate(ppis),
        label=meta["label"].to_numpy(str),
        date=meta["date"].to_numpy(str),
        session=meta["session"].to_numpy(str),
        proto=meta["proto"].to_numpy(str),
        offset=meta["offset"].to_numpy(float),
    )
    print(f"\n저장: {out}")
    # 처음 10초 안에 열린 플로우가 몰려 있으면 캡처를 켤 때의 재연결일 수 있다 → train --skip-first
    summary = meta.groupby(["label", "proto"]).size().unstack(fill_value=0)
    summary["처음 10초"] = meta[meta["offset"] < 10].groupby("label").size()
    print(summary.reindex([label for label in LABELS if label in summary.index]).fillna(0).astype(int))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status", action="store_true", help="파일 이름 검사와 라벨×날짜 파일 수만 보여 준다")
    parser.add_argument("--folder", type=Path, default=SELF_COLLECTED_DIR)
    args = parser.parse_args()

    if args.status:
        valid, invalid = list_pcaps(args.folder)
        print(collection_status(args.folder) if valid else f"{args.folder}에 이름 규칙에 맞는 pcap이 아직 없다.")
        if invalid:
            print(f"\n이름 규칙(라벨_날짜_순번.pcap, 라벨은 {', '.join(LABELS)})에 안 맞는 파일:")
            for p in invalid:
                print(f"  {p.name}")
    else:
        build_self_collected(args.folder)


if __name__ == "__main__":
    main()
