"""데이터셋 불러오기 — 모두 X (N, 3, 30) float32 원시 PPI, y (N,) int64 로 맞춘다.

학습·시험은 항상 날짜 단위로 나눈다 (CESNET은 주차, 직접 수집은 수집 날짜).
condition에는 85% 같은 숫자를 보고할 때 함께 적을 측정 조건을 담는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from src.config import CESNET_DIR, DIR, IPT, PROCESSED_DIR, PROTOS, TASKS


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


def pick_val_sessions(sessions: np.ndarray, labels: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """학습 데이터 중 일부 세션(파일)을 통째로 검증용으로 뺀다. 세션 하나가 양쪽에 나뉘지 않는다.

    라벨마다 따로 세션의 frac만큼(적어도 1개) 고른다. 세션이 1개뿐인 라벨은 학습에 남긴다.
    전체에서 무작위로 고르면 파일이 적을 때 어떤 앱이 검증에서 통째로 빠지고, 검증 macro-F1로 고르는 에포크가 틀어진다
    (11/09 첫 학습: 파일 2개만 뽑혀 유튜브가 빠짐)."""
    val = np.zeros(len(sessions), dtype=bool)
    for label in np.unique(labels):
        mine = labels == label
        unique = np.unique(sessions[mine])
        if len(unique) < 2:
            continue
        n_val = max(1, round(len(unique) * frac))
        val |= mine & np.isin(sessions, rng.choice(unique, n_val, replace=False))
    return val


def skip_packets(X: np.ndarray, n: int) -> np.ndarray:
    """앞 n개 패킷(ClientHello·QUIC Initial 같은 핸드셰이크)을 지우고 나머지를 앞으로 당긴다. 뒤는 0으로 채운다.
    새 첫 패킷의 간격은 0으로 둔다 — 플로우 첫 패킷처럼, 가린 패킷과의 간격도 남기지 않으려고.
    패킷이 n개 이하인 플로우는 빈 입력(모두 0)이 된다. 빼지 않는다 — 시험 세트를 가리지 않은 결과와 같게 두려고."""
    if n <= 0:
        return X
    out = np.zeros_like(X)
    out[:, :, :X.shape[2] - n] = X[:, :, n:]
    out[:, IPT, 0] = 0
    return out


def hide_handshake(data: Data, n: int) -> Data:
    """학습·검증·시험 모두 앞 n개 패킷을 가린다 (skip_packets). 통신 라이브러리 지문(첫 패킷 크기 등) 없이도 맞히는지 보는 실험.
    condition에 skip_packets를 남겨 추론(src.predict)도 같게 가리고, 가린 뒤 빈 입력이 된 플로우 수도 적는다."""
    def hide(split: Split) -> Split:
        return Split(skip_packets(split.X, n), split.y)

    empty = {name: int(((split.X[:, DIR] != 0).sum(1) <= n).sum())
             for name, split in (("train", data.train), ("val", data.val), ("test", data.test))}
    return replace(data, train=hide(data.train), val=hide(data.val), test=hide(data.test),
                   condition={**data.condition, "skip_packets": n, "empty_after_skip": empty})


def split_last_days(dates: list[str], n: int) -> tuple[list[str], list[str]]:
    """기간의 날짜(YYYYMMDD) → (앞 날짜들, 마지막 n일). 시간 순서 검증에 쓴다."""
    dates = sorted(dates)
    if not 0 < n < len(dates):
        raise ValueError(f"검증으로 뗄 날짜 수는 1 이상 {len(dates)} 미만이어야 한다 (기간 {len(dates)}일)")
    return dates[:-n], dates[-n:]


def split_from_df(df) -> Split:
    """datazoo DataFrame → Split (원시 PPI, 앱 번호)."""
    return Split(np.stack(df["PPI"].to_numpy()).astype(np.float32), df["APP"].to_numpy(np.int64))


def load_cesnet(
    size: str = "XS",
    train_period: str = "W-2022-44",
    test_period: str = "W-2022-45",
    train_size: int | None = None,
    test_size: int | None = None,
    val_days: int = 0,
    root: Path = CESNET_DIR,
) -> Data:
    """CESNET-QUIC22 → 주차 단위 학습/시험 (README 3절). 처음 부르면 데이터를 내려받는다.

    PPI는 변환 없이 원시값 (ppi_transform 기본값 None), 라벨은 get_known_apps() 순서의 정수다.
    검증은 학습 주차에서 무작위 20%(datazoo 기본값)다. val_days > 0이면 학습 주차의 마지막 val_days일을 검증으로 뗀다 —
    시간 순서 검증: 검증이 학습보다 뒤라 다음 주 시험과 더 닮는다. 대신 학습 날짜가 그만큼 줄어든다.
    """
    from cesnet_datazoo.config import AppSelection, DatasetConfig, ValidationApproach
    from cesnet_datazoo.datasets import CESNET_QUIC22

    dataset = CESNET_QUIC22(str(root), size=size)
    if val_days:
        train_dates, val_dates = split_last_days(dataset.time_periods[train_period], val_days)
        periods = {"train_period_name": f"{train_period}-first{len(train_dates)}d", "train_dates": train_dates,
                   "val_approach": ValidationApproach.VALIDATION_DATES,
                   "val_period_name": f"{train_period}-last{val_days}d", "val_dates": val_dates}
        val_rule = f"학습 주차의 마지막 {val_days}일 ({', '.join(val_dates)}) — 학습은 앞 {len(train_dates)}일"
    else:
        periods = {"train_period_name": train_period}
        val_rule = "학습 주차에서 무작위 20% (datazoo 기본값)"
    dataset.set_dataset_config_and_initialize(DatasetConfig(
        dataset=dataset,
        apps_selection=AppSelection.ALL_KNOWN,
        test_period_name=test_period,
        train_size=train_size or "all",
        test_known_size=test_size or "all",
        **periods,
    ))

    train, val, test = (split_from_df(dataset.get_train_df()), split_from_df(dataset.get_val_df()),
                        split_from_df(dataset.get_test_df()))
    classes = [str(app) for app in dataset.get_known_apps()]
    return Data(
        train=train, val=val, test=test, classes=classes,
        condition={
            "dataset": f"CESNET-QUIC22 {size}",
            "classes": len(classes),
            "proto": "quic",
            "split": "주차 단위",
            "train_period": train_period,
            "test_period": test_period,
            "val": val_rule,
            **({"val_days": val_days} if val_days else {}),
            "flows": {"train": len(train.y), "val": len(val.y), "test": len(test.y)},
        },
    )


def load_self_collected(
    path: Path = PROCESSED_DIR / "self_collected.npz",
    task: str = "app",
    skip_first_s: float = 0.0,
    proto: str = "all",
    test_dates: list[str] | None = None,
    n_test_dates: int = 3,
    val_frac: float = 0.1,
    seed: int = 0,
) -> Data:
    """src.preprocess가 만든 npz → 날짜 단위 학습/검증/시험.

    task: "app"(앱 분류) 또는 "screen"(인스타 화면 분류). config.TASKS의 라벨만 쓴다.
    skip_first_s: 캡처 시작 뒤 이 시간 안에 열린 플로우를 뺀다 (캡처를 켤 때 몰리는 재연결 제외).
    proto: "all" · "quic" · "tcp" (config.PROTOS).
    test_dates: 시험용 날짜. 없으면 이 과제 데이터의 마지막 n_test_dates일.
    """
    npz = np.load(path)
    keep = np.isin(npz["label"], TASKS[task]) & (npz["offset"] >= skip_first_s) & np.isin(npz["proto"], PROTOS[proto])
    if not keep.any():
        raise ValueError(f"{task} 과제({', '.join(TASKS[task])})에 남는 플로우가 없다 (proto={proto}, skip_first_s={skip_first_s}).")
    X, labels, dates, sessions = (npz[key][keep] for key in ("ppi", "label", "date", "session"))
    # numpy 문자열 대신 str — model.pt를 torch.load 기본값(weights_only)으로 다시 불러올 수 있게
    classes = sorted(set(labels.tolist()))
    y = np.searchsorted(classes, labels).astype(np.int64)

    all_dates = sorted(set(dates.tolist()))
    if test_dates is None:
        test_dates = all_dates[-n_test_dates:]
    train_mask, test_mask = split_by_date(dates, test_dates)
    if not train_mask.any():
        raise ValueError(f"학습할 날짜가 없다 — 수집 날짜 {all_dates}가 모두 시험 날짜다. --test-dates로 시험 날짜를 줄인다.")
    val_mask = np.zeros(len(y), dtype=bool)
    val_mask[train_mask] = pick_val_sessions(sessions[train_mask], y[train_mask], val_frac, np.random.default_rng(seed))
    train_mask &= ~val_mask

    def take(mask: np.ndarray) -> Split:
        return Split(X[mask], y[mask])

    train_dates = sorted(set(dates[train_mask | val_mask].tolist()))
    return Data(
        train=take(train_mask), val=take(val_mask), test=take(test_mask), classes=classes,
        condition={
            "dataset": "self_collected",
            "task": task,
            "classes": len(classes),
            "labels": classes,
            "proto": proto,
            "skip_first_s": skip_first_s,
            "split": "수집 날짜 단위",
            "train_dates": train_dates,
            "test_dates": sorted(test_dates),
            "val": f"학습 날짜 중 라벨마다 세션(파일) {val_frac:.0%} (적어도 1개)",
            "flows": {"train": int(train_mask.sum()), "val": int(val_mask.sum()), "test": int(test_mask.sum())},
        },
    )
