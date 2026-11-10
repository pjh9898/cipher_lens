import json

import pandas as pd
import pytest

from src.report import load_runs, plot_confusion, results_table, to_markdown

CESNET = {"dataset": "CESNET-QUIC22 XS", "classes": 101, "proto": "quic", "split": "주차 단위",
          "train_period": "W-2022-44", "test_period": "W-2022-45", "flows": {"train": 800, "val": 200, "test": 500}}


def self_collected(task, **extra):
    return {"dataset": "self_collected", "task": task, "classes": 3, "proto": "all", "skip_first_s": 0.0,
            "split": "수집 날짜 단위", "train_dates": ["1013", "1014"], "test_dates": ["1015"],
            "flows": {"train": 80, "val": 10, "test": 30}, **extra}


def write_metrics(root, name, condition, model="cnn", accuracy=0.9, val=None, seed=0, history=None):
    run = root / name
    run.mkdir(parents=True)
    metrics = {"accuracy": accuracy, "macro_f1": accuracy - 0.1, "val": val, "condition": condition,
               "args": {"model": model, "seed": seed}}
    if history:
        metrics["history"] = history
    (run / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")


def test_results_table(tmp_path):
    write_metrics(tmp_path, "20261101-000000_self-screen_cnn", self_collected("screen", skip_first_s=10))
    write_metrics(tmp_path, "20261101-000000_self-app_transformer", self_collected("app"), model="transformer")
    write_metrics(tmp_path, "20261101-000000_self-app_cnn", self_collected("app", proto="quic"))
    write_metrics(tmp_path, "20261101-000000_cesnet_cnn", CESNET, accuracy=0.5)
    write_metrics(tmp_path, "20261102-000000_cesnet_cnn", CESNET, accuracy=0.8)  # 같은 조합의 더 최근 결과

    table = results_table(load_runs(tmp_path))
    assert list(zip(table["데이터셋"], table["모델"])) == [
        ("CESNET", "cnn"), ("직접 수집 app", "cnn"), ("직접 수집 app", "transformer"), ("직접 수집 screen", "cnn"),
    ]
    assert table.loc[0, "정확도"] == 0.8  # 가장 최근 결과만
    assert table.loc[0, "분할 (학습 → 시험)"] == "W-2022-44 → W-2022-45"
    assert table.loc[1, "분할 (학습 → 시험)"] == "1013,1014 → 1015"
    assert table.loc[1, "조건"] == "quic"
    assert table.loc[3, "조건"] == "앞 10초 제외"
    assert len(results_table(load_runs(tmp_path), latest_only=False)) == 5

    lines = to_markdown(table).splitlines()
    assert lines[0].startswith("| 데이터셋 | 모델 | 정확도 | macro-F1 | 검증 정확도 | 검증 macro-F1 |")
    assert lines[2].startswith("| CESNET | cnn | 80.00% | 70.00% | — | — | 101 |")  # 검증 점수가 없던 결과는 —
    assert len(lines) == 2 + 4


def test_results_table_keeps_each_condition(tmp_path):
    """같은 데이터셋·모델이어도 조건(앞 패킷 가림 등)이 다르면 최신 결과를 따로 남긴다."""
    write_metrics(tmp_path, "20261101-000000_self-app_cnn", self_collected("app"))
    write_metrics(tmp_path, "20261102-000000_self-app_cnn", self_collected("app", skip_packets=5))
    table = results_table(load_runs(tmp_path))
    assert list(table["조건"]) == ["", "앞 패킷 5개 가림"]


def test_results_table_averages_seeds(tmp_path):
    """조건이 같고 시드만 다른 결과는 한 줄로 묶어 시드 평균 ± 표준편차(%p)를 보인다. 같은 시드를 다시 돌리면 최근 것만 센다.
    검증은 끝 무렵 5 에포크 평균(실험을 비교하는 값)이고, 학습 곡선이 없는 랜덤포레스트는 한 번 잰 검증 점수다."""
    history = [{"epoch": e, "loss": 1.0, "val_accuracy": 0.80 + 0.01 * e, "val_macro_f1": 0.70 + 0.01 * e} for e in range(1, 8)]
    # 끝 5 에포크(3~7) 평균은 0.85 / 0.75 — 가장 좋은 에포크(7: 0.87 / 0.77)가 아니라
    write_metrics(tmp_path, "20261101-000000_cesnet_cnn", CESNET, accuracy=0.5)  # 같은 시드 0을 다시 돌려 덮인다
    write_metrics(tmp_path, "20261102-000000_cesnet_cnn", CESNET, accuracy=0.80,
                  val={"accuracy": 0.87, "macro_f1": 0.77, "epoch": 7}, history=history)
    write_metrics(tmp_path, "20261103-000000_cesnet_cnn", CESNET, accuracy=0.82, seed=1)
    write_metrics(tmp_path, "20261104-000000_cesnet_cnn", CESNET, accuracy=0.84, seed=2)
    write_metrics(tmp_path, "20261105-000000_cesnet_rf", CESNET, model="rf", val={"accuracy": 0.8, "macro_f1": 0.6})
    table = results_table(load_runs(tmp_path))
    assert list(table["조건"]) == ["quic", "quic"]
    assert list(table["시드 수"]) == [3, 1]
    assert table.loc[0, "정확도"] == pytest.approx(0.82) and table.loc[0, "정확도 표준편차"] == pytest.approx(0.02)
    assert table.loc[0, "검증 정확도"] == pytest.approx(0.85)  # 검증 점수가 있는 결과(시드 0)만 평균
    assert table.loc[1, "검증 macro-F1"] == 0.6
    lines = to_markdown(table).splitlines()
    assert lines[0].endswith("| 조건 | 시드 수 | run |") and "표준편차" not in lines[0]
    assert lines[2].startswith("| CESNET | cnn | 82.00% ± 2.00 | 72.00% ± 2.00 | 85.00% | 75.00% | 101 |")
    assert lines[2].endswith("| quic | 3 | 20261104-000000_cesnet_cnn 외 2개 |")
    assert lines[3].startswith("| CESNET | rf | 90.00% | 80.00% | 80.00% | 60.00% | 101 |")
    every = results_table(load_runs(tmp_path), latest_only=False)  # --all: 결과마다 한 줄, 시드 열과 함께
    assert list(every["시드"]) == [0, 0, 1, 2, 0]


def test_empty_runs(tmp_path):
    assert results_table(load_runs(tmp_path)).empty


def test_plot_confusion(tmp_path):
    pd.DataFrame([[5, 1], [2, 4]], index=["webtoon", "youtube"], columns=["webtoon", "youtube"]).to_csv(tmp_path / "confusion.csv")
    out = plot_confusion(tmp_path)
    assert out == tmp_path / "confusion.png"
    assert out.stat().st_size > 0
