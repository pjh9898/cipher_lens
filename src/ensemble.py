"""앙상블 — 학습한 모델 여러 개의 앱별 확률을 평균해 분류한다 (학습 없음 — experiments.md E6 · E7).

    python -m src.ensemble runs/20261110-145453_cesnet_cnn runs/20261110-161733_cesnet_transformer
    python -m src.ensemble --drift runs/20261110-181522_cesnet_cnn     # CESNET 드리프트 트랙 (모델 하나도 된다)

모델마다 검증 · 시험 세트의 확률을 한 번씩 구해, 모델 하나씩과 앙상블(2개 이상일 때, 모두의 평균)의 정확도 · macro-F1과
처리량(시험 세트, 모델만 — 앙상블은 모델별 추론 시간의 합)을 마크다운 표로 보인다. runs/에는 남기지 않는다.

데이터는 첫 모델의 학습 인자(metrics.json의 args)대로 다시 만든다. 모델마다 데이터 조건이 다르면 멈춘다 —
직접 수집은 시드가 검증 세션을 정해서, 시드가 다른 모델을 섞으면 검증 세트에 다른 모델의 학습 데이터가 들어간다.
CESNET의 검증 분할은 시드와 상관없다 (datazoo 기본값). 앞 패킷 가리기(skip_packets)는 모델마다 추론 때 따로 한다 —
가린 모델과 안 가린 모델을 섞을 수 있다.

--drift: CESNET 모델을 드리프트 트랙(config.DRIFT_*, experiments.md A8)으로 다시 잰다 — 검증은 구글 인증서가 바뀐 뒤
(W-2022-45의 11/11~13), 시험은 W-2022-46 · 47. 모델은 다시 학습하지 않는다.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data import Data, load_cesnet_drift, skip_packets
from src.predict import Classifier, load_classifier, predict_proba
from src.report import to_markdown
from src.train import load_data

# 데이터를 정하는 학습 인자와 기본값 — 예전 학습의 args에 없는 인자는 기본값으로 본다.
# skip_packets는 넣지 않는다 — 데이터는 그대로 불러오고 모델마다 추론 때 가린다 (predict_all).
DATA_ARGS = {"dataset": None, "cesnet_size": "XS", "train_size": None, "task": "app", "skip_first": 0.0, "proto": "all",
             "test_dates": None, "val_days": 0}
CHUNK = 200_000  # 확률을 이만큼씩 구해 더한다 — CESNET 시험(255만 × 앱 101)을 한꺼번에 들고 있지 않으려고


def data_args(metrics: dict) -> dict:
    """모델을 학습한 데이터를 정하는 인자. 직접 수집은 시드가 검증 세션을 정해서 시드도 넣는다."""
    args = metrics["args"]
    picked = {k: args.get(k, default) for k, default in DATA_ARGS.items()}
    if picked["dataset"] == "self":
        picked["seed"] = args.get("seed", 0)
    return picked


def hidden_packets(clf: Classifier) -> int:
    """학습 때 가린 앞 패킷 수 — 추론 때도 똑같이 가린다."""
    return int(clf.metrics.get("args", {}).get("skip_packets") or 0)


def predict_all(clfs: list[Classifier], X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """모델마다의 예측 (모델 수, N), 앙상블(확률 평균)의 예측 (N,), 모델마다 추론 시간(초).
    X는 가리지 않은 원시 PPI다. 모델마다 학습 때처럼 앞 패킷을 가려 넣는다 (가리는 시간도 추론 시간에 넣는다)."""
    singles = np.zeros((len(clfs), len(X)), dtype=np.int64)
    ensemble = np.zeros(len(X), dtype=np.int64)
    seconds = np.zeros(len(clfs))
    for start in range(0, len(X), CHUNK):
        chunk = X[start:start + CHUNK]
        total = 0.0
        for i, clf in enumerate(clfs):
            began = time.perf_counter()
            proba = predict_proba(clf.model, skip_packets(chunk, hidden_packets(clf)), clf.device)
            seconds[i] += time.perf_counter() - began
            singles[i, start:start + len(chunk)] = proba.argmax(1)
            total = total + proba  # 합의 argmax = 평균의 argmax
        ensemble[start:start + len(chunk)] = total.argmax(1)
    return singles, ensemble, seconds


def scores(y: np.ndarray, pred: np.ndarray, n_classes: int) -> tuple[float | None, float | None]:
    """(정확도, macro-F1). 빈 세트는 (None, None)."""
    if len(y) == 0:
        return None, None
    f1 = f1_score(y, pred, labels=np.arange(n_classes), average="macro", zero_division=0)
    return float((pred == y).mean()), float(f1)


def evaluate_ensemble(clfs: list[Classifier], names: list[str], data: Data, prefix: str = "") -> pd.DataFrame:
    """모델 하나씩과 앙상블(모델이 2개 이상일 때)의 검증 · 시험 점수, 처리량(시험 세트, 모델만).
    prefix는 점수 열 이름 앞에 붙인다 (드리프트 트랙 — 「드리프트 검증 정확도」)."""
    n = len(data.classes)
    val_single, val_ensemble, _ = predict_all(clfs, data.val.X)
    test_single, test_ensemble, seconds = predict_all(clfs, data.test.X)
    preds = [(name, val_single[i], test_single[i], seconds[i]) for i, name in enumerate(names)]
    if len(clfs) > 1:
        preds.append((f"앙상블 ({len(clfs)}개)", val_ensemble, test_ensemble, seconds.sum()))
    rows = []
    for name, val_pred, test_pred, secs in preds:
        val_accuracy, val_macro_f1 = scores(data.val.y, val_pred, n)
        test_accuracy, test_macro_f1 = scores(data.test.y, test_pred, n)
        rows.append({
            "모델": name,
            f"{prefix}검증 정확도": val_accuracy,
            f"{prefix}검증 macro-F1": val_macro_f1,
            f"{prefix}시험 정확도": test_accuracy,
            f"{prefix}시험 macro-F1": test_macro_f1,
            "처리량 (초당 플로우)": f"{len(data.test.y) / secs:,.0f}" if secs > 0 else "—",
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="학습 결과 폴더 (model.pt가 있는 것). 2개 이상이면 앙상블 줄도 보인다")
    parser.add_argument("--drift", action="store_true", help="CESNET 드리프트 트랙으로 잰다 (검증 W-2022-45의 11/11~13 · 시험 W-2022-46 · 47)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    clfs = [load_classifier(run, args.device) for run in args.runs]
    first = data_args(clfs[0].metrics)
    if args.drift:
        # 드리프트 트랙은 학습 데이터와 상관없이 같은 검증 · 시험을 쓴다 — CESNET 모델이면 섞어도 된다
        for run, clf in zip(args.runs, clfs):
            if data_args(clf.metrics)["dataset"] != "cesnet":
                parser.error(f"{run.name}: --drift는 CESNET 모델에만 쓴다")
            if data_args(clf.metrics)["cesnet_size"] != first["cesnet_size"]:
                parser.error(f"{run.name}: CESNET 크기가 첫 모델과 다르다")
        data = load_cesnet_drift(size=first["cesnet_size"])
    else:
        for run, clf in zip(args.runs[1:], clfs[1:]):
            if data_args(clf.metrics) != first:
                parser.error(f"{run.name}: 데이터 조건이 첫 모델과 다르다 — {data_args(clf.metrics)} vs {first}")
        data = load_data({**clfs[0].metrics["args"], "skip_packets": 0})  # 가리기는 모델마다 predict_all에서
    for run, clf in zip(args.runs, clfs):
        if clf.classes != list(data.classes):
            parser.error(f"{run.name}: 앱 목록이 데이터와 다르다")
    if args.drift:
        print(f"드리프트 트랙 — 검증: {data.condition['val']} · 시험: {data.condition['test_period']}")
    print(f"검증 {len(data.val.y):,}개 · 시험 {len(data.test.y):,}개 · 앱 {len(data.classes)}개\n")
    names = [run.name for run in args.runs]
    print(to_markdown(evaluate_ensemble(clfs, names, data, prefix="드리프트 " if args.drift else "")))


if __name__ == "__main__":
    main()
