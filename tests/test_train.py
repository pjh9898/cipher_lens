import functools
import json
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
from src.train import class_weights, evaluate, fit, fit_forest, predict


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
    assert clf.metrics["val"]["epoch"] == 1 and 0 <= clf.metrics["val"]["macro_f1"] <= 1  # 고른 에포크의 검증 점수
    assert [r["epoch"] for r in clf.metrics["history"]] == [1]  # 학습 곡선
    table = results_table(load_runs(tmp_path / "runs"))
    assert list(zip(table["데이터셋"], table["모델"])) == [("직접 수집 app", "transformer")]


def test_fit_records_history_and_validation_score():
    """에포크마다 학습 곡선을 남기고, 결과의 검증 점수는 fit이 고른 에포크(검증 macro-F1 최고, 같으면 앞 에포크)의 값이다."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    data = Data(toy_split(256, rng), toy_split(64, rng), toy_split(64, rng), classes=["small", "large"])
    history = []
    model = fit(build_model("cnn", 2), data, epochs=3, batch_size=32, lr=1e-3, device="cpu", history=history)
    assert [r["epoch"] for r in history] == [1, 2, 3]
    best = max(history, key=lambda r: r["val_macro_f1"])
    val = train.validation_score(model, data, history, "cpu")
    assert val == {"accuracy": best["val_accuracy"], "macro_f1": best["val_macro_f1"], "epoch": best["epoch"]}
    assert evaluate(model, data.val, 2, "cpu")["macro_f1"] == pytest.approx(val["macro_f1"])  # 돌려준 가중치가 그 에포크의 것


def test_fit_cosine_schedule_lowers_lr():
    """cosine이면 학습률이 에포크마다 코사인 곡선으로 준다 (3 에포크: 1e-3 → 7.5e-4 → 2.5e-4), 기본값은 고정."""
    rng = np.random.default_rng(0)
    data = Data(toy_split(128, rng), toy_split(32, rng), toy_split(32, rng), classes=["small", "large"])
    for schedule, expected in (("cosine", [1e-3, 7.5e-4, 2.5e-4]), ("constant", [1e-3] * 3)):
        history = []
        fit(build_model("cnn", 2), data, epochs=3, batch_size=32, lr=1e-3, device="cpu", history=history, schedule=schedule)
        assert [r["lr"] for r in history] == pytest.approx(expected)


def test_main_cosine_schedule_is_marked_in_report(tmp_path, npz_path, monkeypatch):
    """--lr-schedule cosine은 학습 곡선의 학습률과 결과 표 「조건」에 남는다."""
    monkeypatch.setattr(train, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(train, "load_self_collected", functools.partial(load_self_collected, npz_path))
    monkeypatch.setattr(sys, "argv", ["train", "--dataset", "self", "--task", "app", "--model", "cnn", "--epochs", "2",
                                      "--device", "cpu", "--test-dates", "1017", "--lr-schedule", "cosine"])
    train.main()

    (run,) = (tmp_path / "runs").iterdir()
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert [r["lr"] for r in metrics["history"]] == pytest.approx([1e-3, 5e-4])
    assert list(load_runs(tmp_path / "runs")["조건"]) == ["코사인 학습률"]


def test_main_width_is_saved_and_marked_in_report(tmp_path, npz_path, monkeypatch):
    """--width는 model.pt에 남아 다시 불러올 때(src.predict) 같은 폭으로 만들고, 결과 표 「조건」에도 남는다."""
    monkeypatch.setattr(train, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(train, "load_self_collected", functools.partial(load_self_collected, npz_path))
    monkeypatch.setattr(sys, "argv", ["train", "--dataset", "self", "--task", "app", "--model", "cnn", "--epochs", "1",
                                      "--device", "cpu", "--test-dates", "1017", "--width", "16"])
    train.main()

    (run,) = list_runs(tmp_path / "runs")
    clf = load_classifier(run, device="cpu")  # 폭을 모르면 가중치 모양이 맞지 않아 여기서 실패한다
    assert clf.model.features[0][0].out_channels == 16
    assert list(load_runs(tmp_path / "runs")["조건"]) == ["폭 16"]


def test_main_rejects_width_for_forest(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train", "--dataset", "self", "--model", "rf", "--width", "128"])
    with pytest.raises(SystemExit):
        train.main()


def test_class_weights_balance_classes():
    """개수가 적은 앱일수록 가중치가 크고, 앱마다 (개수 × 가중치)가 같다. 학습에 없는 앱은 0."""
    y = np.array([0] * 6 + [1] * 2)
    weights = class_weights(y, n_classes=3)
    assert weights[1] > weights[0] > 0
    assert weights[2] == 0
    counts = np.bincount(y, minlength=3)
    assert np.allclose(counts[:2] * weights[:2], len(y) / 2)


def imbalanced_toy_data(rng: np.random.Generator) -> Data:
    """클래스 1이 학습에 16개뿐 (클래스 0의 1/15)."""
    def take(split: Split, n_small: int) -> Split:
        keep = np.flatnonzero(split.y == 0).tolist() + np.flatnonzero(split.y == 1)[:n_small].tolist()
        return Split(split.X[keep], split.y[keep])
    return Data(take(toy_split(512, rng), 16), toy_split(64, rng), toy_split(64, rng), classes=["small", "large"])


@pytest.mark.parametrize("name", MODELS)
def test_fit_with_class_weight(name):
    """가중치를 켠 학습이 돌고 드문 클래스도 맞힌다. 가짜 데이터가 쉬워서 가중치를 꺼도 맞히므로,
    가중치의 효과가 아니라 경로만 확인한다 (가중치 식은 test_class_weights_balance_classes)."""
    torch.manual_seed(0)
    data = imbalanced_toy_data(np.random.default_rng(0))
    model = fit(build_model(name, 2), data, epochs=3, batch_size=32, lr=1e-3, device="cpu", balanced=True)
    assert evaluate(model, data.test, 2, "cpu")["per_class"]["recall"][1] > 0.9


def test_forest_learns_toy_data():
    rng = np.random.default_rng(0)
    data = Data(toy_split(256, rng), toy_split(64, rng), toy_split(64, rng), classes=["small", "large"])
    forest = fit_forest(data, seed=0, max_samples=100)
    result = evaluate(forest, data.test, 2, "cpu")
    assert result["accuracy"] > 0.9
    assert result["confusion"].shape == (2, 2)


def test_forest_predicts_in_chunks():
    """랜덤포레스트를 나눠서 예측해도 한 번에 예측한 것과 같다."""
    rng = np.random.default_rng(0)
    data = Data(toy_split(256, rng), toy_split(64, rng), toy_split(64, rng), classes=["small", "large"])
    forest = fit_forest(data, seed=0, max_samples=100)
    whole = forest.predict(data.test.X.reshape(len(data.test.X), -1))
    np.testing.assert_array_equal(predict(forest, data.test.X, "cpu", rf_batch_size=7), whole)
    assert len(predict(forest, data.test.X[:0], "cpu")) == 0


def test_main_forest_baseline(tmp_path, npz_path, monkeypatch):
    """rf는 결과 표에만 들어가고, model.pt가 없어 프로토타입 목록에는 나오지 않는다."""
    monkeypatch.setattr(train, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(train, "load_self_collected", functools.partial(load_self_collected, npz_path))
    monkeypatch.setattr(sys, "argv", ["train", "--dataset", "self", "--task", "app", "--model", "rf", "--test-dates", "1017"])
    train.main()

    (run,) = (tmp_path / "runs").iterdir()
    assert {p.name for p in run.iterdir()} == {"metrics.json", "per_class.csv", "confusion.csv", "confusion.png"}
    assert list_runs(tmp_path / "runs") == []
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["val"]) == {"accuracy", "macro_f1"} and "history" not in metrics  # 학습 뒤 검증 세트로 잰 값
    table = results_table(load_runs(tmp_path / "runs"))
    assert list(zip(table["데이터셋"], table["모델"])) == [("직접 수집 app", "rf")]


def test_main_skip_packets_is_marked_in_report(tmp_path, npz_path, monkeypatch):
    monkeypatch.setattr(train, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(train, "load_self_collected", functools.partial(load_self_collected, npz_path))
    monkeypatch.setattr(sys, "argv", ["train", "--dataset", "self", "--task", "app", "--model", "rf", "--test-dates", "1017",
                                      "--skip-packets", "2"])
    train.main()
    assert list(load_runs(tmp_path / "runs")["조건"]) == ["앞 패킷 2개 가림"]


def test_main_without_class_weight_is_marked_in_report(tmp_path, npz_path, monkeypatch):
    """--class-weight none으로 돌린 결과는 결과 표 「조건」에 표시된다 (기본값 balanced는 표시하지 않음)."""
    monkeypatch.setattr(train, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(train, "load_self_collected", functools.partial(load_self_collected, npz_path))
    monkeypatch.setattr(sys, "argv", ["train", "--dataset", "self", "--task", "app", "--model", "rf", "--test-dates", "1017",
                                      "--class-weight", "none"])
    train.main()

    runs = load_runs(tmp_path / "runs")
    assert list(runs["조건"]) == ["클래스 가중치 없음"]
