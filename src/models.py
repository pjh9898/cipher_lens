"""분류 모델 — 입력은 원시 PPI (B, 3, 30), 출력은 앱별 점수 (B, 앱 수).

정규화를 모델 안에 넣어 두어서 학습·평가·프로토타입이 모두 같은 원시 PPI를 넣으면 된다.
"""
import torch
from torch import nn

from src.config import DIR, IPT, N_PACKETS, SIZE


class Normalize(nn.Module):
    """간격은 log1p(ms), 크기는 /1500, 방향(±1, 채운 자리 0)은 그대로."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ipt = torch.log1p(x[:, IPT].clamp(min=0))
        return torch.stack([ipt, x[:, DIR], x[:, SIZE] / 1500.0], dim=1)


def conv_block(c_in: int, c_out: int, kernel: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2), nn.BatchNorm1d(c_out), nn.ReLU())


class CNN1D(nn.Module):
    """기본 모델 — 패킷 순서를 따라 1D 합성곱."""

    def __init__(self, n_classes: int, width: int = 64, dropout: float = 0.3):
        super().__init__()
        self.norm = Normalize()
        self.features = nn.Sequential(
            conv_block(3, width, 5),
            conv_block(width, width, 5),
            nn.MaxPool1d(2),
            conv_block(width, 2 * width, 3),
            conv_block(2 * width, 2 * width, 3),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(2 * width, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(self.norm(x)))


class SmallTransformer(nn.Module):
    """비교 모델 — 패킷 하나를 토큰 하나로 보는 작은 Transformer 인코더."""

    def __init__(self, n_classes: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = Normalize()
        self.embed = nn.Linear(3, d_model)
        self.pos = nn.Parameter(torch.randn(1, N_PACKETS, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = x[:, DIR] == 0  # 30개보다 짧은 플로우의 채운 자리
        pad[:, 0] = False  # 빈 플로우가 들어와도 NaN이 나지 않게
        h = self.embed(self.norm(x).transpose(1, 2)) + self.pos
        h = self.encoder(h, src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(-1).float()
        return self.head((h * keep).sum(1) / keep.sum(1))


MODELS = {"cnn": CNN1D, "transformer": SmallTransformer}


def build_model(name: str, n_classes: int) -> nn.Module:
    return MODELS[name](n_classes)
