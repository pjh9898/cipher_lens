"""진짜 tshark로 전처리 전체를 돌려 본다 — tshark가 없으면 건너뛴다 (Wireshark를 설치하면 돈다).

PCAPdroid와 같은 raw IP pcap을 바이트로 직접 만든다. 내용은 conftest.packets와 같은 장면:
TCP 핸드셰이크·재전송·순수 ACK, 캡처 전에 열린 TCP·QUIC 연결, DNS, IPv6 QUIC(long header로 시작).
특히 quic.header_form이 이 tshark 버전에서 제대로 나오는지(todo 2단계) 확인한다.
"""
import socket
import struct

import numpy as np
import pytest

from src.config import DIR, IPT, N_PACKETS, SIZE
from src.preprocess import find_tshark, packets_from_tshark, pcap_to_flows, run_tshark


def has_tshark() -> bool:
    try:
        find_tshark()
        return True
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(not has_tshark(), reason="tshark 없음 — Wireshark를 설치하면 돈다")

TCP, UDP = 6, 17
LINKTYPE_RAW = 101  # PCAPdroid가 쓰는 형식 (이더넷 헤더 없이 IP부터)
EPOCH = 1_760_000_000
PHONE, PHONE6, SERVER6 = "10.0.0.2", "fd00::2", "2001:db8::1"
SYN, ACK, PSH = 0x02, 0x10, 0x08


def checksum(data: bytes) -> int:
    if len(data) % 2:
        data += b"\0"
    total = sum(struct.unpack(f"!{len(data) // 2}H", data))
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return ~total & 0xFFFF


def ip(src: str, dst: str, proto: int, segment: bytes) -> bytes:
    """IPv4·IPv6 패킷. segment는 체크섬 자리를 0으로 둔 TCP·UDP 세그먼트 — 체크섬을 채워 넣는다."""
    v6 = ":" in src
    family = socket.AF_INET6 if v6 else socket.AF_INET
    s, d = socket.inet_pton(family, src), socket.inet_pton(family, dst)
    pseudo = s + d + (struct.pack("!IxxxB", len(segment), proto) if v6 else struct.pack("!xBH", proto, len(segment)))
    at = 16 if proto == TCP else 6
    segment = segment[:at] + struct.pack("!H", checksum(pseudo + segment) or 0xFFFF) + segment[at + 2:]
    if v6:
        return struct.pack("!IHBB", 6 << 28, len(segment), proto, 64) + s + d + segment
    header = struct.pack("!BBHHHBBH", 0x45, 0, 20 + len(segment), 0, 0x4000, 64, proto, 0) + s + d
    return header[:10] + struct.pack("!H", checksum(header)) + header[12:] + segment


def tcp(src, sport, dst, dport, seq, ack, flags, payload=0) -> bytes:
    """payload바이트짜리 TLS 레코드(application data)를 실은 TCP 패킷."""
    data = b"\x17\x03\x03" + (payload - 5).to_bytes(2, "big") + bytes(payload - 5) if payload else b""
    return ip(src, dst, TCP, struct.pack("!HHIIBBHHH", sport, dport, seq, ack, 5 << 4, flags, 65535, 0, 0) + data)


def udp(src, sport, dst, dport, payload: bytes) -> bytes:
    return ip(src, dst, UDP, struct.pack("!HHHH", sport, dport, 8 + len(payload), 0) + payload)


def quic_long(dcid: bytes, scid: bytes, size: int) -> bytes:
    """QUIC v1 Initial (long header). 암호화된 부분은 0 — tshark는 복호화에 실패해도 헤더는 읽는다."""
    head = b"\xc3" + (1).to_bytes(4, "big") + bytes([len(dcid)]) + dcid + bytes([len(scid)]) + scid + b"\x00"
    rest = size - len(head) - 2  # 길이 필드(2바이트 가변 정수) 뒤: 패킷 번호 + 암호문
    return head + (0x4000 | rest).to_bytes(2, "big") + bytes(rest)


def quic_short(dcid: bytes, size: int) -> bytes:
    return b"\x43" + dcid + bytes(size - 1 - len(dcid))


def write_pcap(path, packets: list[tuple[float, bytes]]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, LINKTYPE_RAW))
        for t, data in sorted(packets, key=lambda p: p[0]):
            usec = round(t * 1e6)
            f.write(struct.pack("<IIII", EPOCH + usec // 1_000_000, usec % 1_000_000, len(data), len(data)) + data)


@pytest.fixture
def pcap(tmp_path):
    s = "1.2.3.4"
    packets = [
        # TCP(TLS) 플로우
        (0.000, tcp(PHONE, 50000, s, 443, 1000, 0, SYN)),
        (0.010, tcp(s, 443, PHONE, 50000, 5000, 1001, SYN | ACK)),
        (0.011, tcp(PHONE, 50000, s, 443, 1001, 5001, ACK)),
        (0.020, tcp(PHONE, 50000, s, 443, 1001, 5001, PSH | ACK, 300)),
        (0.050, tcp(s, 443, PHONE, 50000, 5001, 1301, ACK, 1400)),
        (0.051, tcp(s, 443, PHONE, 50000, 5001, 1301, ACK, 1400)),  # 재전송 → 빠진다
        (0.060, tcp(PHONE, 50000, s, 443, 1301, 6401, ACK)),  # 순수 ACK → 빠진다
        (0.070, tcp(s, 443, PHONE, 50000, 6401, 1301, PSH | ACK, 200)),
        # 캡처 전에 열린 TCP 연결 (SYN 없음)
        (0.100, tcp(s, 443, PHONE, 50001, 9000, 7000, PSH | ACK, 500)),
        (0.110, tcp(PHONE, 50001, s, 443, 7000, 9500, PSH | ACK, 100)),
        # DNS
        (0.200, udp(PHONE, 50003, "8.8.8.8", 53, bytes(40))),
        (0.210, udp("8.8.8.8", 53, PHONE, 50003, bytes(80))),
    ]
    # IPv6 QUIC: Initial 2개(long header)로 시작, 나머지는 short header. 35개 → 앞 30개만
    client_cid, server_cid = b"\xc1" * 8, b"\x5e" * 8
    for i in range(35):
        size = 1200 + i
        if i == 0:
            payload = quic_long(b"\xd0" * 8, client_cid, size)
        elif i == 1:
            payload = quic_long(client_cid, server_cid, size)
        else:
            payload = quic_short(server_cid if i % 2 == 0 else client_cid, size)
        src, dst = ((PHONE6, 50002), (SERVER6, 443)) if i % 2 == 0 else ((SERVER6, 443), (PHONE6, 50002))
        packets.append((0.300 + i * 0.001, udp(*src, *dst, payload)))
    # 캡처 전에 열린 QUIC 연결 (short header만)
    for i in range(5):
        src, dst = ((PHONE, 50004), ("5.6.7.8", 443)) if i % 2 == 0 else (("5.6.7.8", 443), (PHONE, 50004))
        packets.append((0.400 + i * 0.001, udp(*src, *dst, quic_short(b"\x77" * 8, 1200 + i))))

    path = tmp_path / "youtube_1013_1.pcap"
    write_pcap(path, packets)
    return path


def test_tshark_fields(pcap):
    pk = packets_from_tshark(run_tshark(pcap))
    assert len(pk) == 12 + 35 + 5
    assert (pk["proto"] == "tcp").sum() == 10
    assert pk["flags"].iloc[0] == SYN
    quic = pk[pk["src"].isin([PHONE6, SERVER6])]
    assert list(quic["payload"]) == list(range(1200, 1235))  # UDP 헤더 8바이트를 뺀 크기
    assert list(quic["quic_long"]) == [True, True] + [False] * 33  # long header = Initial 2개
    assert not pk.loc[pk["sport"].eq(50004) | pk["dport"].eq(50004), "quic_long"].any()


def test_pcap_to_flows(pcap):
    ppi, meta = pcap_to_flows(pcap)
    assert ppi.shape == (2, 3, N_PACKETS)
    assert list(meta["proto"]) == ["tcp", "udp"]
    assert list(meta["label"]) == ["youtube", "youtube"]
    np.testing.assert_allclose(meta["offset"], [0.0, 0.3], atol=1e-6)

    tcp_flow, quic_flow = ppi
    np.testing.assert_array_equal(tcp_flow[SIZE, :4], [300, 1400, 200, 0])  # ACK·재전송 뺌
    np.testing.assert_array_equal(tcp_flow[DIR, :4], [1, -1, -1, 0])
    np.testing.assert_allclose(tcp_flow[IPT, :3], [0, 30, 20], atol=1e-3)
    np.testing.assert_array_equal(quic_flow[SIZE], 1200 + np.arange(30))
    np.testing.assert_array_equal(quic_flow[DIR], np.where(np.arange(30) % 2 == 0, 1, -1))
    np.testing.assert_allclose(quic_flow[IPT, 1:], 1.0, atol=1e-3)


def test_partial_flows_are_detected(pcap):
    _, meta = pcap_to_flows(pcap, server_ports=None, drop_partial=False)
    assert list(meta["starts_with_handshake"]) == [True, False, False, True, False]  # TCP · 중간 TCP · DNS · QUIC · 중간 QUIC
