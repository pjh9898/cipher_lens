import numpy as np
import pandas as pd
import pytest

from src.config import DIR, IPT, N_PACKETS, SIZE
from src.preprocess import flows_from_packets, packets_from_tshark, parse_pcap_name

PHONE = "10.0.0.2"


def tcp(t, src, sport, dst, dport, payload, seq, flags):
    return dict(time=t, src=src, dst=dst, sport=sport, dport=dport, proto="tcp", payload=payload, seq=seq, flags=flags)


def udp(t, src, sport, dst, dport, payload):
    return dict(time=t, src=src, dst=dst, sport=sport, dport=dport, proto="udp", payload=payload, seq=np.nan, flags=np.nan)


@pytest.fixture
def packets() -> pd.DataFrame:
    s = "1.2.3.4"
    rows = [
        # TCP(TLS) 플로우: 핸드셰이크 + 데이터 + 재전송 + 순수 ACK
        tcp(0.000, PHONE, 50000, s, 443, 0, 0, 0x02),  # SYN
        tcp(0.010, s, 443, PHONE, 50000, 0, 0, 0x12),  # SYN-ACK
        tcp(0.011, PHONE, 50000, s, 443, 0, 1, 0x10),  # ACK
        tcp(0.020, PHONE, 50000, s, 443, 300, 1, 0x18),  # ClientHello
        tcp(0.050, s, 443, PHONE, 50000, 1400, 1, 0x10),
        tcp(0.051, s, 443, PHONE, 50000, 1400, 1, 0x10),  # 재전송 → 빠진다
        tcp(0.060, PHONE, 50000, s, 443, 0, 301, 0x10),  # 순수 ACK → 빠진다
        tcp(0.070, s, 443, PHONE, 50000, 200, 1401, 0x18),
        # 캡처 전에 열린 TCP 연결 (SYN 없음) → 기본값으로 빠진다
        tcp(0.100, s, 443, PHONE, 50001, 500, 9000, 0x18),
        tcp(0.110, PHONE, 50001, s, 443, 100, 7000, 0x18),
        # DNS → 443이 아니라 빠진다
        udp(0.200, PHONE, 50003, "8.8.8.8", 53, 40),
        udp(0.210, "8.8.8.8", 53, PHONE, 50003, 80),
    ]
    # QUIC 플로우: 35개 패킷 → 앞 30개만
    rows += [
        udp(0.300 + i * 0.001, *((PHONE, 50002, "5.6.7.8", 443) if i % 2 == 0 else ("5.6.7.8", 443, PHONE, 50002)), 1200 + i)
        for i in range(35)
    ]
    return pd.DataFrame(rows)


def test_flows_shape_and_filters(packets):
    ppi, meta = flows_from_packets(packets)
    assert ppi.shape == (2, 3, N_PACKETS)
    assert ppi.dtype == np.float32
    assert list(meta["proto"]) == ["tcp", "udp"]
    assert list(meta["ppi_len"]) == [3, 30]


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


def test_filters_can_be_turned_off(packets):
    _, meta = flows_from_packets(packets, server_ports=None, drop_partial_tcp=False)
    assert len(meta) == 4
    assert set(meta["server_port"]) == {443, 53, 50001}  # SYN이 없으면 첫 패킷을 보낸 서버가 클라이언트로 잡힌다


def test_empty_input():
    ppi, meta = flows_from_packets(packets_from_tshark(""))
    assert ppi.shape == (0, 3, N_PACKETS)
    assert len(meta) == 0


def test_packets_from_tshark():
    csv = (
        "frame.time_epoch,ip.src,ip.dst,ipv6.src,ipv6.dst,tcp.srcport,tcp.dstport,udp.srcport,udp.dstport,"
        "tcp.len,udp.length,tcp.seq,tcp.flags\n"
        "1760000000.000000,10.0.0.2,1.2.3.4,,,50000,443,,,0,,0,0x0002\n"
        "1760000000.100000,,,fd00::2,2001:db8::1,,,50002,443,,1208,,\n"
    )
    pk = packets_from_tshark(csv)
    assert list(pk["proto"]) == ["tcp", "udp"]
    assert list(pk["src"]) == ["10.0.0.2", "fd00::2"]
    assert list(pk["dport"]) == [443, 443]
    assert list(pk["payload"]) == [0, 1200]  # UDP 헤더 8바이트 제외
    assert pk["flags"].iloc[0] == 0x02


@pytest.mark.parametrize("name, expected", [
    ("youtube_1013_1.pcap", {"app": "youtube", "date": "1013", "session": "youtube_1013_1"}),
    ("kakaotalk_1101_12.pcapng", {"app": "kakaotalk", "date": "1101", "session": "kakaotalk_1101_12"}),
    ("YouTube_1013_1.pcap", None),  # 대문자
    ("PCAPdroid_13_Oct_15_30_00.pcap", None),  # PCAPdroid 기본 이름
    ("youtube_1013.pcap", None),  # 순번 없음
])
def test_parse_pcap_name(name, expected):
    assert parse_pcap_name(name) == expected
