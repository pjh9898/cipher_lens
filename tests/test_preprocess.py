import numpy as np
import pytest

import src.preprocess as preprocess
from src.config import DIR, IPT, N_PACKETS, SIZE
from src.preprocess import build_self_collected, flows_from_packets, packets_from_tshark, parse_pcap_name


def test_flows_shape_and_filters(packets):
    ppi, meta = flows_from_packets(packets)
    assert ppi.shape == (2, 3, N_PACKETS)
    assert ppi.dtype == np.float32
    assert list(meta["proto"]) == ["tcp", "udp"]
    assert list(meta["ppi_len"]) == [3, 30]


def test_offset_is_seconds_from_capture_start(packets):
    _, meta = flows_from_packets(packets)
    np.testing.assert_allclose(meta["offset"], [0.0, 0.3])


def test_tcp_flow_skips_acks_and_retransmissions(packets):
    ppi, _ = flows_from_packets(packets)
    tcp_flow = ppi[0]
    np.testing.assert_array_equal(tcp_flow[SIZE, :3], [300, 1400, 200])
    np.testing.assert_array_equal(tcp_flow[DIR, :3], [1, -1, -1])
    np.testing.assert_allclose(tcp_flow[IPT, :3], [0, 30, 20], atol=1e-3)  # ms, 재전송 뺀 뒤의 간격
    assert not tcp_flow[:, 3:].any()  # 나머지는 0으로 채움


def test_quic_flow_truncated_to_30(packets):
    ppi, _ = flows_from_packets(packets)
    quic = ppi[1]
    np.testing.assert_array_equal(quic[SIZE], 1200 + np.arange(30))
    np.testing.assert_array_equal(quic[DIR], np.where(np.arange(30) % 2 == 0, 1, -1))
    assert quic[IPT, 0] == 0
    np.testing.assert_allclose(quic[IPT, 1:], 1.0, atol=1e-3)


def test_ppi_does_not_depend_on_ip_or_port(packets):
    """IP·포트를 모두 바꿔도 모델 입력은 같다 — 모델이 주소로 답을 외울 수 없다 (CLAUDE.md 모델 입력 규칙)."""
    ppi, meta = flows_from_packets(packets)
    ips = {ip: f"192.0.2.{i + 1}" for i, ip in enumerate(sorted(set(packets["src"]) | set(packets["dst"])))}
    moved = packets.assign(
        src=packets["src"].map(ips),
        dst=packets["dst"].map(ips),
        sport=packets["sport"].where(packets["sport"] == 443, packets["sport"] + 7000),  # 서버 포트(443)만 남긴다
        dport=packets["dport"].where(packets["dport"] == 443, packets["dport"] + 7000),
    )
    moved_ppi, _ = flows_from_packets(moved)
    np.testing.assert_array_equal(ppi, moved_ppi)
    assert ppi.shape[1] == 3  # 간격·방향·크기 세 줄뿐


def test_filters_can_be_turned_off(packets):
    _, meta = flows_from_packets(packets, server_ports=None, drop_partial=False)
    assert len(meta) == 5
    assert list(meta["starts_with_handshake"]) == [True, False, False, True, False]
    assert set(meta["server_port"]) == {443, 53, 50001}  # SYN이 없으면 첫 패킷을 보낸 서버가 클라이언트로 잡힌다


def test_empty_input():
    ppi, meta = flows_from_packets(packets_from_tshark(""))
    assert ppi.shape == (0, 3, N_PACKETS)
    assert len(meta) == 0


def test_packets_from_tshark():
    csv = (
        "frame.time_epoch,ip.src,ip.dst,ipv6.src,ipv6.dst,tcp.srcport,tcp.dstport,udp.srcport,udp.dstport,"
        "tcp.len,udp.length,tcp.seq,tcp.flags,quic.header_form\n"
        "1760000000.000000,10.0.0.2,1.2.3.4,,,50000,443,,,0,,0,0x0002,\n"
        "1760000000.100000,,,fd00::2,2001:db8::1,,,50002,443,,1208,,,1\n"
        "1760000000.200000,,,fd00::2,2001:db8::1,,,50002,443,,108,,,False\n"
    )
    pk = packets_from_tshark(csv)
    assert list(pk["proto"]) == ["tcp", "udp", "udp"]
    assert list(pk["src"]) == ["10.0.0.2", "fd00::2", "fd00::2"]
    assert list(pk["dport"]) == [443, 443, 443]
    assert list(pk["payload"]) == [0, 1200, 100]  # UDP 헤더 8바이트 제외
    assert pk["flags"].iloc[0] == 0x02
    assert list(pk["quic_long"]) == [False, True, False]


@pytest.fixture
def collected(tmp_path, packets, monkeypatch):
    """직접 수집 폴더 — tshark 대신 파일마다 conftest의 가짜 패킷이 나온다."""
    folder = tmp_path / "self_collected"
    folder.mkdir()
    for name in ["youtube_1013_1.pcap", "instagram-reels_1014_1.pcap", "PCAPdroid_13_Oct.pcap"]:
        (folder / name).touch()
    monkeypatch.setattr(preprocess, "run_tshark", lambda pcap: "")
    monkeypatch.setattr(preprocess, "packets_from_tshark", lambda csv_text: packets.copy())
    return folder


def test_build_self_collected(tmp_path, collected, capsys):
    """npz에 data.load_self_collected가 읽는 키가 다 들어가는지."""
    out = build_self_collected(collected, tmp_path / "out.npz")
    npz = np.load(out)
    assert set(npz.files) == {"ppi", "label", "date", "session", "proto", "offset"}
    assert npz["ppi"].shape == (4, 3, N_PACKETS)  # 이름 규칙에 맞는 파일 2개 × 플로우 2개
    assert sorted(set(npz["label"])) == ["instagram-reels", "youtube"]
    # 443 패킷 50개 중 중간부터 잡힌 TCP 2개 + QUIC 5개
    assert "youtube_1013_1.pcap: 플로우 2개 (중간부터 잡혀 뺀 연결 2개 = 443 패킷의 14%)\n" in capsys.readouterr().out


def test_warns_when_app_was_already_running(tmp_path, collected, capsys, monkeypatch):
    """앱 분류 파일만 경고한다 — 화면 분류는 앱을 켠 채 캡처하므로 중간 연결이 많은 게 정상이다."""
    monkeypatch.setattr(preprocess, "PARTIAL_WARN", 0.1)
    build_self_collected(collected, tmp_path / "out.npz")
    lines = capsys.readouterr().out.splitlines()
    assert next(line for line in lines if line.startswith("youtube_")).endswith("강제 종료하지 않고 캡처를 켠 것 같다")
    assert "강제 종료" not in next(line for line in lines if line.startswith("instagram-reels_"))


@pytest.mark.parametrize("name, expected", [
    ("youtube_1013_1.pcap", {"label": "youtube", "date": "1013", "session": "youtube_1013_1"}),
    ("instagram-reels_1013_2.pcap", {"label": "instagram-reels", "date": "1013", "session": "instagram-reels_1013_2"}),
    ("webtoon_1101_12.pcapng", {"label": "webtoon", "date": "1101", "session": "webtoon_1101_12"}),
    ("instgram-reels_1013_1.pcap", None),  # 라벨 오타
    ("kakaotalk_1013_1.pcap", None),  # 분류 대상이 아님
    ("YouTube_1013_1.pcap", None),  # 대문자
    ("PCAPdroid_13_Oct_15_30_00.pcap", None),  # PCAPdroid 기본 이름
    ("youtube_1013.pcap", None),  # 순번 없음
])
def test_parse_pcap_name(name, expected):
    assert parse_pcap_name(name) == expected
