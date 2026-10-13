"""학습·평가 — 정확도 + macro-F1 + 혼동 행렬을 runs/ 아래에 남긴다.

    python -m src.train --dataset cesnet --model cnn
    python -m src.train --dataset self --model transformer --test-dates 1027,1028
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import RUNS_DIR
from src.data import Data, Split, load_cesnet, load_self_collected
from src.models import MODELS, build_model


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: str, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    preds = [
        model(torch.from_numpy(X[i:i + batch_size]).to(device)).argmax(1).cpu().numpy()
        for i in range(0, len(X), batch_size)
    ]
    return np.concatenate(preds) if preds else np.empty(0, dtype=np.int64)


def evaluate(model: nn.Module, split: Split, n_classes: int, device: str) -> dict:
    pred = predict(model, split.X, device)
    labels = np.arange(n_classes)
    return {
        "accuracy": float(accuracy_score(split.y, pred)),
        "macro_f1": float(f1_score(split.y, pred, labels=labels, average="macro", zero_division=0)),
        "confusion": confusion_matrix(split.y, pred, labels=labels),
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
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cesnet-size", default="XS", choices=["XS", "S", "M", "L"])
    parser.add_argument("--train-size", type=int, default=None, help="CESNET 학습 샘플 수 제한 (빠른 실험용)")
    parser.add_argument("--test-dates", default=None, help="직접 수집: 시험용 날짜 (예: 1027,1028). 없으면 마지막 3일")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "cesnet":
        data = load_cesnet(size=args.cesnet_size, train_size=args.train_size)
    else:
        test_dates = args.test_dates.split(",") if args.test_dates else None
        data = load_self_collected(test_dates=test_dates, seed=args.seed)
    print(json.dumps(data.condition, ensure_ascii=False, indent=2))

    model = build_model(args.model, len(data.classes)).to(args.device)
    started = time.perf_counter()
    model = fit(model, data, args.epochs, args.batch_size, args.lr, args.device)
    train_seconds = time.perf_counter() - started

    test = evaluate(model, data.test, len(data.classes), args.device)
    print(f"\n시험  정확도 {test['accuracy']:.4f}  macro-F1 {test['macro_f1']:.4f}")

    out = RUNS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{args.dataset}_{args.model}"
    out.mkdir(parents=True)
    metrics = {
        "accuracy": test["accuracy"],
        "macro_f1": test["macro_f1"],
        "condition": data.condition,
        "args": vars(args),
        "train_seconds": round(train_seconds, 1),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(test["confusion"], index=data.classes, columns=data.classes).to_csv(out / "confusion.csv")
    torch.save({"model": args.model, "classes": data.classes, "state_dict": model.state_dict()}, out / "model.pt")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
