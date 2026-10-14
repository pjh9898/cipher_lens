"""여러 테스트가 같이 쓰는 가짜 데이터."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.models import build_model

PHONE = "10.0.0.2"


def tcp(t, src, sport, dst, dport, payload, seq, flags):
    return dict(time=t, src=src, dst=dst, sport=sport, dport=dport, proto="tcp", payload=payload, seq=seq, flags=flags,
                quic_long=False)


def udp(t, src, sport, dst, dport, payload, quic_long=False):
    return dict(time=t, src=src, dst=dst, sport=sport, dport=dport, proto="udp", payload=payload, seq=np.nan, flags=np.nan,
                quic_long=quic_long)


def quic_flow(start, port, n, first_long):
    """n개 패킷, 짝수 번째는 폰→서버. first_long이면 핸드셰이크(long header)부터 잡힌 연결."""
    return [
        udp(start + i * 0.001, *((PHONE, port, "5.6.7.8", 443) if i % 2 == 0 else ("5.6.7.8", 443, PHONE, port)),
            1200 + i, quic_long=first_long and i < 2)
        for i in range(n)
    ]


@pytest.fixture
def packets() -> pd.DataFrame:
    """preprocess.packets_from_tshark가 내놓는 모양의 패킷 표 — 남는 플로우는 TCP 1개(t=0) + QUIC 1개(t=0.3)."""
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
    rows += quic_flow(0.300, 50002, 35, first_long=True)  # 35개 → 앞 30개만
    rows += quic_flow(0.400, 50004, 5, first_long=False)  # 캡처 전에 열린 QUIC 연결 → 기본값으로 빠진다
    return pd.DataFrame(rows)


@pytest.fixture
def npz_path(tmp_path):
    """preprocess가 만드는 npz 모양. 라벨 4개(앱 2 + 화면 2) × 5일 × 하루 2세션 × 세션당 플로우 4개
    (offset 0, 5, 20, 60초 / proto는 udp, tcp, udp, tcp 순)."""
    labels, dates, sessions, offsets, protos = [], [], [], [], []
    for label in ["youtube", "instagram", "instagram-reels", "instagram-feed"]:
        for day in ["1013", "1014", "1015", "1016", "1017"]:
            for seq in [1, 2]:
                for offset, proto in zip([0.0, 5.0, 20.0, 60.0], ["udp", "tcp", "udp", "tcp"]):
                    labels.append(label)
                    dates.append(day)
                    sessions.append(f"{label}_{day}_{seq}")
                    offsets.append(offset)
                    protos.append(proto)
    n = len(labels)
    path = tmp_path / "self_collected.npz"
    np.savez(path, ppi=np.random.default_rng(0).random((n, 3, 30), dtype=np.float32), label=np.array(labels),
             date=np.array(dates), session=np.array(sessions), proto=np.array(protos), offset=np.array(offsets))
    return path


@pytest.fixture
def make_run(tmp_path):
    """src.train이 남기는 모양의 학습 결과 폴더 (model.pt + metrics.json). 가중치는 학습하지 않은 값."""

    def make(name="20261101-120000_self-app_cnn", condition=None, classes=("instagram", "webtoon", "youtube"), model="cnn"):
        run = tmp_path / "runs" / name
        run.mkdir(parents=True)
        state = build_model(model, len(classes)).state_dict()
        torch.save({"model": model, "classes": list(classes), "state_dict": state}, run / "model.pt")
        metrics = {"accuracy": 0.9, "macro_f1": 0.8, "condition": condition or {}, "args": {"model": model}}
        (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        return run

    return make
