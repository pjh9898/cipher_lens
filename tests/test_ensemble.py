import functools
import json
import sys

import numpy as np
import pytest
import torch

import src.ensemble as ensemble
import src.train as train
from src.data import load_self_collected
from src.models import build_model
from src.predict import Classifier, predict_proba


def test_predict_all_averages_probabilities_in_chunks(monkeypatch):
    """앙상블 예측은 확률 평균의 argmax이고, 나눠 구해도(CHUNK) 한꺼번에 구한 것과 같다."""
    monkeypatch.setattr(ensemble, "CHUNK", 7)
    torch.manual_seed(0)
    clfs = [Classifier(build_model(name, 3).eval(), ["a", "b", "c"], {}, "cpu") for name in ("cnn", "transformer")]
    X = np.random.default_rng(0).random((20, 3, 30), dtype=np.float32) * 1000
    singles, combined, seconds = ensemble.predict_all(clfs, X)
    probas = [predict_proba(c.model, X, "cpu") for c in clfs]
    assert singles.shape == (2, 20)
    assert all((singles[i] == p.argmax(1)).all() for i, p in enumerate(probas))
    assert (combined == np.mean(probas, axis=0).argmax(1)).all()
    assert (seconds > 0).all()


def train_run(tmp_path, npz_path, monkeypatch, *extra):
    monkeypatch.setattr(train, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(train, "load_self_collected", functools.partial(load_self_collected, npz_path))
    monkeypatch.setattr(sys, "argv", ["train", "--dataset", "self", "--task", "app", "--epochs", "1", "--device", "cpu",
                                      "--test-dates", "1017", *extra])
    train.main()


def test_main_end_to_end(tmp_path, npz_path, monkeypatch, capsys):
    """학습한 모델 둘 → 학습 때 조건대로 데이터를 다시 만들어, 모델마다와 앙상블의 점수를 표로 보인다.
    모델 하나씩의 시험 정확도가 저장값과 같아야 데이터를 똑같이 다시 만든 것이다."""
    train_run(tmp_path, npz_path, monkeypatch, "--model", "cnn")
    train_run(tmp_path, npz_path, monkeypatch, "--model", "transformer")
    runs = sorted((tmp_path / "runs").iterdir())
    capsys.readouterr()

    monkeypatch.setattr(sys, "argv", ["ensemble", *map(str, runs), "--device", "cpu"])
    ensemble.main()
    lines = capsys.readouterr().out.splitlines()
    assert "| 모델 | 검증 정확도 | 검증 macro-F1 | 시험 정확도 | 시험 macro-F1 | 처리량 (초당 플로우) |" in lines
    for run in runs:
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        (row,) = [line for line in lines if line.startswith(f"| {run.name} |")]
        assert f"| {metrics['accuracy']:.2%} |" in row
    assert any(line.startswith("| 앙상블 (2개) |") for line in lines)


def test_main_rejects_models_with_different_data(tmp_path, npz_path, monkeypatch):
    """직접 수집은 시드가 검증 세션을 정한다 — 시드가 다른 모델을 섞으면 검증 점수가 섞이므로 멈춘다."""
    train_run(tmp_path, npz_path, monkeypatch, "--model", "cnn")
    train_run(tmp_path, npz_path, monkeypatch, "--model", "transformer", "--seed", "1")
    runs = sorted((tmp_path / "runs").iterdir())
    monkeypatch.setattr(sys, "argv", ["ensemble", *map(str, runs), "--device", "cpu"])
    with pytest.raises(SystemExit):
        ensemble.main()


def test_data_args_fill_defaults_for_old_runs():
    """예전 학습의 args에 없는 인자는 기본값으로 본다. CESNET은 시드가 검증 분할을 바꾸지 않아 시드를 보지 않는다."""
    old = {"args": {"dataset": "cesnet", "model": "cnn", "seed": 0}}
    new = {"args": {"dataset": "cesnet", "model": "transformer", "seed": 2, "cesnet_size": "XS", "skip_packets": 0,
                    "lr_schedule": "cosine"}}
    assert ensemble.data_args(old) == ensemble.data_args(new)
    assert ensemble.data_args({"args": {"dataset": "self", "seed": 1}})["seed"] == 1
    # 검증을 다르게 뗀 모델(시간 순서 검증)은 섞지 않는다
    assert ensemble.data_args({"args": {**new["args"], "val_days": 1}}) != ensemble.data_args(new)

