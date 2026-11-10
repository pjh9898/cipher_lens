import pytest
import torch

from src.config import DIR, IPT, N_PACKETS, SIZE
from src.models import MODELS, build_model


def random_ppi(batch: int, lengths: list[int]) -> torch.Tensor:
    x = torch.zeros(batch, 3, N_PACKETS)
    for i, n in enumerate(lengths):
        x[i, IPT, 1:n] = torch.rand(max(n - 1, 0)) * 100
        x[i, DIR, :n] = torch.where(torch.rand(n) > 0.5, 1.0, -1.0)
        x[i, SIZE, :n] = torch.randint(1, 1500, (n,)).float()
    return x


@pytest.mark.parametrize("name", MODELS)
def test_forward_shape(name):
    model = build_model(name, n_classes=5).eval()
    x = random_ppi(4, [30, 10, 1, 0])  # 마지막은 빈 플로우
    out = model(x)
    assert out.shape == (4, 5)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", MODELS)
def test_width_changes_model_size(name):
    """width는 모델 폭만 바꾼다 — 넓히면 파라미터가 늘고, 입출력 모양은 같다. 기본값은 64."""
    def n_params(model):
        return sum(p.numel() for p in model.parameters())

    default, wide = build_model(name, 5), build_model(name, 5, width=128)
    assert n_params(build_model(name, 5, width=64)) == n_params(default)
    assert n_params(wide) > 2 * n_params(default)
    assert wide.eval()(random_ppi(2, [30, 5])).shape == (2, 5)


def test_transformer_ignores_padding():
    """채운 자리(0)는 결과에 영향을 주지 않는다."""
    torch.manual_seed(0)
    model = build_model("transformer", n_classes=3).eval()
    x = random_ppi(1, [10])
    noisy = x.clone()
    noisy[0, IPT, 10:] = 999  # 방향이 0인 자리의 값만 바꾼다
    torch.testing.assert_close(model(x), model(noisy))
