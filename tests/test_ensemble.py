import dataclasses
import functools
import json
import sys

import numpy as np
import pytest
import torch
from torch import nn

import src.ensemble as ensemble
import src.train as train
from src.config import DIR, SIZE
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


class FirstPacketSize(nn.Module):
    """첫 패킷이 500바이트를 넘으면 1번 앱, 아니면 0번 앱 — 앞 패킷을 가렸는지가 예측에 그대로 드러난다."""

    def forward(self, x):
        return nn.functional.one_hot((x[:, SIZE, 0] > 500).long(), 3).float() * 10


def test_predict_all_hides_packets_per_model():
    """가린 모델과 안 가린 모델을 섞어도, 모델마다 학습 때처럼 앞 패킷을 가린 입력을 받는다."""
    X = np.zeros((4, 3, 30), dtype=np.float32)
    X[:, DIR, :6] = 1
    X[:, SIZE, 0], X[:, SIZE, 5] = 1000, 100  # 5개를 가리면 6번째(100바이트)가 첫 패킷이 된다
    plain = Classifier(FirstPacketSize(), ["a", "b", "c"], {"args": {"skip_packets": 0}}, "cpu")
    hidden = Classifier(FirstPacketSize(), ["a", "b", "c"], {"args": {"skip_packets": 5}}, "cpu")
    singles, _, _ = ensemble.predict_all([plain, hidden], X)
    assert (singles[0] == 1).all()
    assert (singles[1] == 0).all()


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
    # 앞 패킷을 가린 수만 다른 모델은 섞을 수 있다 — 가리기는 모델마다 추론 때 한다
    assert ensemble.data_args({"args": {**new["args"], "skip_packets": 8}}) == ensemble.data_args(new)


def as_cesnet(run):
    """가짜 직접 수집 모델을 CESNET 모델로 꾸민다 (드리프트 트랙은 CESNET 모델만 받는다)."""
    path = run / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics["args"]["dataset"] = "cesnet"
    path.write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")


def test_main_drift_rejects_self_collected(tmp_path, npz_path, monkeypatch):
    train_run(tmp_path, npz_path, monkeypatch, "--model", "cnn")
    (run,) = (tmp_path / "runs").iterdir()
    monkeypatch.setattr(sys, "argv", ["ensemble", "--drift", str(run), "--device", "cpu"])
    with pytest.raises(SystemExit):
        ensemble.main()


def test_main_drift_scores_on_drift_sets(tmp_path, npz_path, monkeypatch, capsys):
    """--drift는 학습 데이터 대신 드리프트 트랙의 검증 · 시험으로 잰다 (가짜 데이터로 바꿔 끼움). 모델 하나도 된다."""
    train_run(tmp_path, npz_path, monkeypatch, "--model", "cnn")
    train_run(tmp_path, npz_path, monkeypatch, "--model", "transformer", "--seed", "1")
    runs = sorted((tmp_path / "runs").iterdir())
    for run in runs:
        as_cesnet(run)
    fake = dataclasses.replace(load_self_collected(npz_path, task="app", test_dates=["1017"]),
                               condition={"val": "드리프트 뒤", "test_period": "W-2022-46 · W-2022-47"})
    sizes = []
    monkeypatch.setattr(ensemble, "load_cesnet_drift", lambda size: sizes.append(size) or fake)
    capsys.readouterr()

    monkeypatch.setattr(sys, "argv", ["ensemble", "--drift", *map(str, runs), "--device", "cpu"])
    ensemble.main()  # 시드가 달라도 멈추지 않는다 — 드리프트 트랙은 학습 데이터와 상관없이 같은 세트를 쓴다
    lines = capsys.readouterr().out.splitlines()
    assert sizes == ["XS"]
    assert "| 모델 | 드리프트 검증 정확도 | 드리프트 검증 macro-F1 | 드리프트 시험 정확도 | 드리프트 시험 macro-F1 | 처리량 (초당 플로우) |" in lines
    assert any(line.startswith("| 앙상블 (2개) |") for line in lines)

    monkeypatch.setattr(sys, "argv", ["ensemble", "--drift", str(runs[0]), "--device", "cpu"])
    ensemble.main()
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith(f"| {runs[0].name} |") for line in lines)
    assert not any(line.startswith("| 앙상블") for line in lines)
