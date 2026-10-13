import numpy as np
import pytest
import torch

from src.config import DIR, SIZE
from src.data import Data, Split
from src.models import MODELS, build_model
from src.train import evaluate, fit


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
