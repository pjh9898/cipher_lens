"""pcap → 플로우마다 앞 30개 패킷의 (간격, 방향, 크기) 행렬.

CESNET-QUIC22의 PPI와 같은 모양 (N, 3, 30)으로 만든다.
- 행 순서: IPT(ms, 첫 패킷은 0) · DIR(+1 클라이언트→서버, −1 서버→클라이언트) · SIZE(L4 페이로드 바이트)
- 30개보다 짧은 플로우는 0으로 채운다.
- TCP는 CESNET과 같게 페이로드가 0인 패킷(순수 ACK 등)과 SEQ가 늘지 않는 패킷(재전송)을 뺀다.
  QUIC(UDP)은 모두 넣는다.
- 클라이언트 = 플로우의 첫 패킷을 보낸 쪽.

IP·포트는 플로우를 묶고 거르는 데만 쓴다. 모델 입력(ppi)에는 넣지 않는다.

    python -m src.preprocess --status   # 파일 이름 검사 + 앱×날짜 파일 수 (tshark 불필요)
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

from src.config import DIR, IPT, N_PACKETS, PROCESSED_DIR, SELF_COLLECTED_DIR, SIZE

TSHARK_FIELDS = [
    "frame.time_epoch",
    "ip.src", "ip.dst", "ipv6.src", "ipv6.dst",
    "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport",
    "tcp.len", "udp.length", "tcp.seq", "tcp.flags",
]
# ICMP 오류 메시지 안에 든 TCP·UDP 헤더는 뺀다
TSHARK_FILTER = "(tcp or udp) and not icmp and not icmpv6"

TCP_SYN = 0x02
TCP_ACK = 0x10

# 직접 수집 파일 이름: 앱_날짜_순번.pcap (예: youtube_1013_1.pcap)
PCAP_NAME = re.compile(r"^(?P<app>[a-z0-9-]+)_(?P<date>\d{4})_(?P<seq>\d+)\.pcap(ng)?$")

PACKET_COLUMNS = ["time", "src", "dst", "sport", "dport", "proto", "payload", "seq", "flags"]


def parse_pcap_name(name: str) -> dict | None:
    """'youtube_1013_1.pcap' → {'app': 'youtube', 'date': '1013', 'session': 'youtube_1013_1'}. 규칙에 안 맞으면 None."""
    m = PCAP_NAME.match(name)
    if m is None:
        return None
    return {"app": m["app"], "date": m["date"], "session": Path(name).stem}


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
    """tshark CSV → 패킷 표 (time, src, dst, sport, dport, proto, payload, seq, flags)."""
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
    })


def flows_from_packets(
    pk: pd.DataFrame,
    server_ports: tuple[int, ...] | None = (443,),
    drop_partial_tcp: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """패킷 표 → (ppi (N, 3, 30) float32, 플로우 정보 표).

    server_ports: 이 포트로 접속한 플로우만 남긴다 (443 = TLS·QUIC). None이면 모두.
    drop_partial_tcp: SYN으로 시작하지 않는 TCP 플로우(캡처 전에 열린 연결)를 뺀다.
        첫 패킷이 없으면 클라이언트·방향을 잘못 잡고 앞 30개 패킷도 진짜 시작이 아니다.
    """
    if pk.empty:
        empty_meta = pd.DataFrame(columns=["proto", "server_port", "start", "starts_with_syn", "packets", "ppi_len"])
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
    meta = pd.DataFrame({
        "proto": first["proto"],
        "server_port": first["dport"],
        "start": first["time"],
        "starts_with_syn": ((flags & TCP_SYN) > 0) & ((flags & TCP_ACK) == 0),
        "packets": pk.groupby("flow").size(),
    })
    keep = pd.Series(True, index=meta.index)
    if server_ports is not None:
        keep &= meta["server_port"].isin(server_ports)
    if drop_partial_tcp:
        keep &= (meta["proto"] != "tcp") | meta["starts_with_syn"]

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
    """pcap 하나 → (ppi, 플로우 정보). 파일 이름이 규칙에 맞으면 app·date·session 열을 붙인다."""
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
    """앱 × 날짜 파일 수 표."""
    valid, _ = list_pcaps(folder)
    names = pd.DataFrame([parse_pcap_name(p.name) for p in valid], columns=["app", "date", "session"])
    return pd.crosstab(names["app"], names["date"], margins=True, margins_name="합계")


def build_self_collected(folder: Path = SELF_COLLECTED_DIR, out: Path = PROCESSED_DIR / "self_collected.npz") -> Path:
    """직접 수집 pcap 전체 → npz (ppi, app, date, session, proto)."""
    valid, invalid = list_pcaps(folder)
    for p in invalid:
        print(f"건너뜀 (이름 규칙 위반): {p.name}")
    ppis, metas = [], []
    for p in valid:
        ppi, meta = pcap_to_flows(p)
        print(f"{p.name}: 플로우 {len(meta)}개")
        ppis.append(ppi)
        metas.append(meta)
    if not metas:
        raise SystemExit(f"{folder}에 처리할 pcap이 없다.")
    meta = pd.concat(metas, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        ppi=np.concatenate(ppis),
        app=meta["app"].to_numpy(str),
        date=meta["date"].to_numpy(str),
        session=meta["session"].to_numpy(str),
        proto=meta["proto"].to_numpy(str),
    )
    print(f"\n저장: {out}")
    print(meta.groupby(["app", "proto"]).size().unstack(fill_value=0))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status", action="store_true", help="파일 이름 검사와 앱×날짜 파일 수만 보여 준다")
    parser.add_argument("--folder", type=Path, default=SELF_COLLECTED_DIR)
    args = parser.parse_args()

    if args.status:
        valid, invalid = list_pcaps(args.folder)
        print(collection_status(args.folder) if valid else f"{args.folder}에 이름 규칙에 맞는 pcap이 아직 없다.")
        if invalid:
            print("\n이름 규칙(앱_날짜_순번.pcap)에 안 맞는 파일:")
            for p in invalid:
                print(f"  {p.name}")
    else:
        build_self_collected(args.folder)


if __name__ == "__main__":
    main()
