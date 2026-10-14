import json

import pandas as pd

from src.report import load_runs, plot_confusion, results_table, to_markdown

CESNET = {"dataset": "CESNET-QUIC22 XS", "classes": 101, "proto": "quic", "split": "주차 단위",
          "train_period": "W-2022-44", "test_period": "W-2022-45", "flows": {"train": 800, "val": 200, "test": 500}}


def self_collected(task, **extra):
    return {"dataset": "self_collected", "task": task, "classes": 3, "proto": "all", "skip_first_s": 0.0,
            "split": "수집 날짜 단위", "train_dates": ["1013", "1014"], "test_dates": ["1015"],
            "flows": {"train": 80, "val": 10, "test": 30}, **extra}


def write_metrics(root, name, condition, model="cnn", accuracy=0.9):
    run = root / name
    run.mkdir(parents=True)
    metrics = {"accuracy": accuracy, "macro_f1": accuracy - 0.1, "condition": condition, "args": {"model": model}}
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
    assert lines[0].startswith("| 데이터셋 | 모델 | 정확도 | macro-F1 |")
    assert lines[2].startswith("| CESNET | cnn | 80.00% | 70.00% | 101 |")
    assert len(lines) == 2 + 4


def test_empty_runs(tmp_path):
    assert results_table(load_runs(tmp_path)).empty


def test_plot_confusion(tmp_path):
    pd.DataFrame([[5, 1], [2, 4]], index=["webtoon", "youtube"], columns=["webtoon", "youtube"]).to_csv(tmp_path / "confusion.csv")
    out = plot_confusion(tmp_path)
    assert out == tmp_path / "confusion.png"
    assert out.stat().st_size > 0
