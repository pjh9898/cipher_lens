import functools
import sys

import numpy as np
import pytest
import torch

import src.train as train
from src.config import DIR, SIZE
from src.data import Data, Split, load_self_collected
from src.models import MODELS, build_model
from src.predict import list_runs, load_classifier
from src.report import load_runs, results_table
from src.train import evaluate, fit, fit_forest


def toy_split(n: int, rng: np.random.Generator) -> Split:
    """클래스 0은 작은 패킷 위주, 클래스 1은 큰 패킷 위주 — 쉽게 나뉘는 가짜 데이터."""
    y = rng.integers(0, 2, n)
    X = np.zeros((n, 3, 30), dtype=np.float32)
    X[:, DIR] = rng.choice([-1, 1], (n, 30))
    X[:, SIZE] = np.where(y[:, None] == 0, rng.integers(50, 200, (n, 30)), rng.integers(1000, 1400, (n, 30)))
    return Split(X, y.astype(np.int64))


@pytest.mark.parametrize("name", MODELS)
def test_fit_learns_toy_data(name):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    data = Data(toy_split(256, rng), toy_split(64, rng), toy_split(64, rng), classes=["small", "large"])
    model = fit(build_model(name, 2), data, epochs=3, batch_size=32, lr=1e-3, device="cpu")
    result = evaluate(model, data.test, 2, "cpu")
    assert result["accuracy"] > 0.9
    assert result["confusion"].shape == (2, 2)
    assert list(result["per_class"].columns) == ["precision", "recall", "f1", "support"]


def test_main_end_to_end(tmp_path, npz_path, monkeypatch):
    """가짜 npz로 학습 → runs/에 결과 → 프로토타입처럼 모델을 다시 불러오기 → 결과 표."""
    monkeypatch.setattr(train, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(train, "load_self_collected", functools.partial(load_self_collected, npz_path))
    monkeypatch.setattr(sys, "argv", ["train", "--dataset", "self", "--task", "app", "--model", "transformer",
                                      "--epochs", "1", "--device", "cpu", "--test-dates", "1017"])
    train.main()

    (run,) = list_runs(tmp_path / "runs")
    assert {p.name for p in run.iterdir()} == {"metrics.json", "per_class.csv", "confusion.csv", "confusion.png", "model.pt"}
    clf = load_classifier(run, device="cpu")  # torch.load 기본값(weights_only)으로 불러와진다
    assert clf.classes == ["instagram", "youtube"]
    assert clf.condition["test_dates"] == ["1017"]
    assert clf.metrics["infer_flows_per_s"] > 0
    table = results_table(load_runs(tmp_path / "runs"))
    assert list(zip(table["데이터셋"], table["모델"])) == [("직접 수집 app", "transformer")]


def test_forest_learns_toy_data():
    rng = np.random.default_rng(0)
    data = Data(toy_split(256, rng), toy_split(64, rng), toy_split(64, rng), classes=["small", "large"])
    forest = fit_forest(data, seed=0, max_samples=100)
    result = evaluate(forest, data.test, 2, "cpu")
    assert result["accuracy"] > 0.9
    assert result["confusion"].shape == (2, 2)


def test_main_forest_baseline(tmp_path, npz_path, monkeypatch):
    """rf는 결과 표에만 들어가고, model.pt가 없어 프로토타입 목록에는 나오지 않는다."""
    monkeypatch.setattr(train, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(train, "load_self_collected", functools.partial(load_self_collected, npz_path))
    monkeypatch.setattr(sys, "argv", ["train", "--dataset", "self", "--task", "app", "--model", "rf", "--test-dates", "1017"])
    train.main()

    (run,) = (tmp_path / "runs").iterdir()
    assert {p.name for p in run.iterdir()} == {"metrics.json", "per_class.csv", "confusion.csv", "confusion.png"}
    assert list_runs(tmp_path / "runs") == []
    table = results_table(load_runs(tmp_path / "runs"))
    assert list(zip(table["데이터셋"], table["모델"])) == [("직접 수집 app", "rf")]
