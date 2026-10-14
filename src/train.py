"""학습·평가 — 정확도 + macro-F1 + 혼동 행렬을 runs/ 아래에 남긴다.

    python -m src.train --dataset cesnet --model cnn
    python -m src.train --dataset self --task app --model transformer --test-dates 1027,1028
    python -m src.train --dataset self --task screen --model cnn --skip-first 10
    python -m src.train --dataset self --task app --model cnn --proto quic
    python -m src.train --dataset cesnet --model rf          # 기준선 (랜덤포레스트)

runs/<시각>_<데이터>_<모델>/ 에 metrics.json(정확도·macro-F1·측정 조건) · per_class.csv · confusion.csv/png · model.pt
(rf는 결과 표용 기준선이라 model.pt를 남기지 않는다 — 프로토타입에서는 고를 수 없다)
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import PROTOS, RUNS_DIR, TASKS
from src.data import Data, Split, load_cesnet, load_self_collected
from src.models import MODELS, build_model
from src.report import plot_confusion

Model = nn.Module | RandomForestClassifier


def fit_forest(data: Data, seed: int, max_samples: int = 200_000) -> RandomForestClassifier:
    """기준선 — 원시 PPI 90개 값(간격·방향·크기 × 30)을 그대로 넣은 랜덤포레스트. 검증 세트는 쓰지 않는다.
    나무마다 학습 플로우를 최대 max_samples개만 뽑는다 (CESNET 학습 주차는 수백만 개라 그대로는 너무 크다)."""
    forest = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, max_samples=min(max_samples, len(data.train.y)),
                                    n_jobs=-1, random_state=seed)
    return forest.fit(data.train.X.reshape(len(data.train.X), -1), data.train.y)


@torch.no_grad()
def predict(model: Model, X: np.ndarray, device: str, batch_size: int = 4096) -> np.ndarray:
    if isinstance(model, RandomForestClassifier):
        return model.predict(X.reshape(len(X), -1))
    model.eval()
    preds = [
        model(torch.from_numpy(X[i:i + batch_size]).to(device)).argmax(1).cpu().numpy()
        for i in range(0, len(X), batch_size)
    ]
    return np.concatenate(preds) if preds else np.empty(0, dtype=np.int64)


def evaluate(model: Model, split: Split, n_classes: int, device: str) -> dict:
    pred = predict(model, split.X, device)
    labels = np.arange(n_classes)
    precision, recall, f1, support = precision_recall_fscore_support(split.y, pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(split.y, pred)),
        "macro_f1": float(f1_score(split.y, pred, labels=labels, average="macro", zero_division=0)),
        "confusion": confusion_matrix(split.y, pred, labels=labels),
        "per_class": pd.DataFrame({"precision": precision, "recall": recall, "f1": f1, "support": support}),
    }


def fit(model: nn.Module, data: Data, epochs: int, batch_size: int, lr: float, device: str) -> nn.Module:
    """검증 macro-F1이 가장 좋았던 에포크의 가중치로 돌려준다."""
    loader = DataLoader(TensorDataset(torch.from_numpy(data.train.X), torch.from_numpy(data.train.y)),
                        batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_f1, best_state = -1.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(yb)
            n += len(yb)

        line = f"epoch {epoch:3d}  loss {total / n:.4f}"
        if len(data.val.y):
            val = evaluate(model, data.val, len(data.classes), device)
            line += f"  val acc {val['accuracy']:.4f}  macro-F1 {val['macro_f1']:.4f}"
            if val["macro_f1"] > best_f1:
                best_f1, best_state = val["macro_f1"], copy.deepcopy(model.state_dict())
        print(line)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["cesnet", "self"], required=True)
    parser.add_argument("--model", choices=[*MODELS, "rf"], required=True, help="rf = 랜덤포레스트 기준선")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cesnet-size", default="XS", choices=["XS", "S", "M", "L"])
    parser.add_argument("--train-size", type=int, default=None, help="CESNET 학습 샘플 수 제한 (빠른 실험용)")
    parser.add_argument("--task", choices=list(TASKS), default="app", help="직접 수집: app(앱 분류) 또는 screen(인스타 화면 분류)")
    parser.add_argument("--skip-first", type=float, default=0.0, help="직접 수집: 캡처 시작 뒤 이 초 안에 열린 플로우 제외")
    parser.add_argument("--proto", choices=list(PROTOS), default="all", help="직접 수집: all · quic · tcp")
    parser.add_argument("--test-dates", default=None, help="직접 수집: 시험용 날짜 (예: 1027,1028). 없으면 마지막 3일")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "cesnet":
        data = load_cesnet(size=args.cesnet_size, train_size=args.train_size)
    else:
        test_dates = args.test_dates.split(",") if args.test_dates else None
        data = load_self_collected(task=args.task, skip_first_s=args.skip_first, proto=args.proto, test_dates=test_dates,
                                   seed=args.seed)
    print(json.dumps(data.condition, ensure_ascii=False, indent=2))

    started = time.perf_counter()
    if args.model == "rf":
        model = fit_forest(data, args.seed)
    else:
        model = fit(build_model(args.model, len(data.classes)).to(args.device), data, args.epochs, args.batch_size, args.lr,
                    args.device)
    train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    test = evaluate(model, data.test, len(data.classes), args.device)
    infer_seconds = time.perf_counter() - started
    print(f"\n시험  정확도 {test['accuracy']:.4f}  macro-F1 {test['macro_f1']:.4f}")
    per_class = test["per_class"].set_axis(data.classes)
    print(per_class.round(3).to_string())

    dataset_name = args.dataset if args.dataset == "cesnet" else f"self-{args.task}"
    out = RUNS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{dataset_name}_{args.model}"
    out.mkdir(parents=True)
    metrics = {
        "accuracy": test["accuracy"],
        "macro_f1": test["macro_f1"],
        "condition": data.condition,
        "args": vars(args),
        "train_seconds": round(train_seconds, 1),
        # 모델만의 처리량 (tshark·플로우 묶기 제외). pcap 전체의 처리량은 python -m src.predict
        "infer_flows_per_s": round(len(data.test.y) / infer_seconds),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    per_class.to_csv(out / "per_class.csv")
    pd.DataFrame(test["confusion"], index=data.classes, columns=data.classes).to_csv(out / "confusion.csv")
    plot_confusion(out)
    if isinstance(model, nn.Module):
        torch.save({"model": args.model, "classes": data.classes, "state_dict": model.state_dict()}, out / "model.pt")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
