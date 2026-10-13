"""데이터셋 불러오기 — 모두 X (N, 3, 30) float32 원시 PPI, y (N,) int64 로 맞춘다.

학습·시험은 항상 날짜 단위로 나눈다 (CESNET은 주차, 직접 수집은 수집 날짜).
condition에는 85% 같은 숫자를 보고할 때 함께 적을 측정 조건을 담는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.config import CESNET_DIR, PROCESSED_DIR


@dataclass
class Split:
    X: np.ndarray
    y: np.ndarray


@dataclass
class Data:
    train: Split
    val: Split
    test: Split
    classes: list[str]
    condition: dict = field(default_factory=dict)


def split_by_date(dates: np.ndarray, test_dates: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(학습 마스크, 시험 마스크). 같은 날짜가 양쪽에 들어가지 않는다."""
    unknown = set(test_dates) - set(dates)
    if unknown:
        raise ValueError(f"데이터에 없는 시험 날짜: {sorted(unknown)}")
    test = np.isin(dates, test_dates)
    return ~test, test


def pick_val_sessions(sessions: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """학습 데이터 중 일부 세션(파일)을 통째로 검증용으로 뺀다. 세션 하나가 양쪽에 나뉘지 않는다."""
    unique = np.unique(sessions)
    if len(unique) < 2:
        return np.zeros(len(sessions), dtype=bool)
    n_val = max(1, round(len(unique) * frac))
    return np.isin(sessions, rng.choice(unique, n_val, replace=False))


def load_cesnet(
    size: str = "XS",
    train_period: str = "W-2022-44",
    test_period: str = "W-2022-45",
    train_size: int | None = None,
    test_size: int | None = None,
    root: Path = CESNET_DIR,
) -> Data:
    """CESNET-QUIC22 → 주차 단위 학습/시험 (README 3절). 처음 부르면 데이터를 내려받는다.

    PPI는 변환 없이 원시값 (ppi_transform 기본값 None), 라벨은 get_known_apps() 순서의 정수다.
    """
    from cesnet_datazoo.config import AppSelection, DatasetConfig
    from cesnet_datazoo.datasets import CESNET_QUIC22

    dataset = CESNET_QUIC22(str(root), size=size)
    dataset.set_dataset_config_and_initialize(DatasetConfig(
        dataset=dataset,
        apps_selection=AppSelection.ALL_KNOWN,
        train_period_name=train_period,
        test_period_name=test_period,
        train_size=train_size or "all",
        test_known_size=test_size or "all",
    ))

    def take(df) -> Split:
        return Split(np.stack(df["PPI"].to_numpy()).astype(np.float32), df["APP"].to_numpy(np.int64))

    train, val, test = take(dataset.get_train_df()), take(dataset.get_val_df()), take(dataset.get_test_df())
    classes = dataset.get_known_apps()
    return Data(
        train=train, val=val, test=test, classes=classes,
        condition={
            "dataset": f"CESNET-QUIC22 {size}",
            "classes": len(classes),
            "split": "주차 단위",
            "train_period": train_period,
            "test_period": test_period,
            "val": "학습 주차에서 무작위 20% (datazoo 기본값)",
            "flows": {"train": len(train.y), "val": len(val.y), "test": len(test.y)},
        },
    )


def load_self_collected(
    path: Path = PROCESSED_DIR / "self_collected.npz",
    test_dates: list[str] | None = None,
    n_test_dates: int = 3,
    val_frac: float = 0.1,
    seed: int = 0,
) -> Data:
    """src.preprocess가 만든 npz → 날짜 단위 학습/검증/시험. test_dates가 없으면 마지막 n_test_dates일."""
    npz = np.load(path)
    X, apps, dates, sessions = npz["ppi"], npz["app"], npz["date"], npz["session"]
    classes = sorted(set(apps))
    y = np.searchsorted(classes, apps).astype(np.int64)

    all_dates = sorted(set(dates))
    if test_dates is None:
        test_dates = all_dates[-n_test_dates:]
    train_mask, test_mask = split_by_date(dates, test_dates)
    val_mask = np.zeros(len(y), dtype=bool)
    val_mask[train_mask] = pick_val_sessions(sessions[train_mask], val_frac, np.random.default_rng(seed))
    train_mask &= ~val_mask

    def take(mask: np.ndarray) -> Split:
        return Split(X[mask], y[mask])

    train_dates = sorted(set(dates[train_mask | val_mask]))
    return Data(
        train=take(train_mask), val=take(val_mask), test=take(test_mask), classes=classes,
        condition={
            "dataset": "self_collected",
            "classes": len(classes),
            "split": "수집 날짜 단위",
            "train_dates": train_dates,
            "test_dates": sorted(test_dates),
            "val": f"학습 날짜 중 세션(파일) {val_frac:.0%}",
            "flows": {"train": int(train_mask.sum()), "val": int(val_mask.sum()), "test": int(test_mask.sum())},
        },
    )
