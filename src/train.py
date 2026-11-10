"""학습·평가 — 정확도 + macro-F1 + 혼동 행렬을 runs/ 아래에 남긴다.

    python -m src.train --dataset cesnet --model cnn --class-weight none --epochs 40
    python -m src.train --dataset cesnet --model cnn --class-weight none --epochs 40 --lr-schedule cosine
    python -m src.train --dataset cesnet --model cnn --class-weight none --epochs 40 --lr-schedule cosine --width 128
    python -m src.train --dataset self --task app --model transformer --test-dates 1027,1028
    python -m src.train --dataset self --task screen --model cnn --skip-first 10
    python -m src.train --dataset self --task app --model cnn --proto quic
    python -m src.train --dataset cesnet --model rf --class-weight none   # 기준선 (랜덤포레스트)
    python -m src.train --dataset self --model cnn --class-weight none   # 앱 간 개수 차이를 맞추지 않고
    python -m src.train --dataset self --model cnn --skip-packets 5      # 앞 패킷 5개(핸드셰이크)를 가리고

runs/<시각>_<데이터>_<모델>/ 에 metrics.json(시험 정확도·macro-F1, 검증 점수, 측정 조건, 학습 곡선) · per_class.csv ·
confusion.csv/png · model.pt
(rf는 결과 표용 기준선이라 model.pt를 남기지 않는다 — 프로토타입에서는 고를 수 없다)

앱마다 플로우 수가 다르다 (직접 수집은 웹툰이 인스타·유튜브의 약 4배). 기본값(--class-weight balanced)은
개수가 적은 앱의 플로우에 더 큰 가중치를 줘서, 모델이 많은 앱만 찍어도 손실이 줄어드는 일을 막는다.
CESNET은 --class-weight none으로 돌린다 — 앱이 101개라 가중치가 0.15~233배로 벌어져, 드문 앱의 정밀도가 무너진다.
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

from src.config import N_PACKETS, PROTOS, RUNS_DIR, TASKS
from src.data import Data, Split, hide_handshake, load_cesnet, load_self_collected
from src.models import MODELS, build_model
from src.report import plot_confusion

Model = nn.Module | RandomForestClassifier


def class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """앱마다 학습 플로우 수에 반비례하는 가중치 = 전체 플로우 수 / (학습에 있는 앱 수 × 그 앱 플로우 수).
    sklearn의 class_weight="balanced"와 같은 식이라, 앱마다 (플로우 수 × 가중치)의 합이 같아진다. 학습에 없는 앱은 0."""
    counts = np.bincount(y, minlength=n_classes)
    present = counts > 0
    weights = np.zeros(n_classes, dtype=np.float32)
    weights[present] = len(y) / (present.sum() * counts[present])
    return weights


def fit_forest(data: Data, seed: int, max_samples: int = 200_000, balanced: bool = True) -> RandomForestClassifier:
    """기준선 — 원시 PPI 90개 값(간격·방향·크기 × 30)을 그대로 넣은 랜덤포레스트. 검증 세트는 쓰지 않는다.
    나무마다 학습 플로우를 최대 max_samples개만 뽑는다 (CESNET 학습 주차는 수백만 개라 그대로는 너무 크다).
    balanced면 신경망과 같은 식(class_weights)으로 앱 간 개수 차이를 맞춘다."""
    forest = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, max_samples=min(max_samples, len(data.train.y)),
                                    class_weight="balanced" if balanced else None, n_jobs=-1, random_state=seed)
    return forest.fit(data.train.X.reshape(len(data.train.X), -1), data.train.y)


@torch.no_grad()
def predict(model: Model, X: np.ndarray, device: str, batch_size: int = 4096, rf_batch_size: int = 100_000) -> np.ndarray:
    """랜덤포레스트도 나눠서 예측한다 — sklearn은 나무마다 (플로우 수 × 앱 수) 확률 배열을 코어 수만큼 동시에 만들어서,
    CESNET 시험 주차(약 280만 플로우 × 앱 105개)를 한 번에 넣으면 메모리가 모자란다."""
    if isinstance(model, RandomForestClassifier):
        chunks = (X[i:i + rf_batch_size] for i in range(0, len(X), rf_batch_size))
        preds = [model.predict(chunk.reshape(len(chunk), -1)) for chunk in chunks]
    else:
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


def fit(model: nn.Module, data: Data, epochs: int, batch_size: int, lr: float, device: str,
        balanced: bool = True, history: list[dict] | None = None, schedule: str = "constant") -> nn.Module:
    """검증 macro-F1이 가장 좋았던 에포크의 가중치로 돌려준다.
    balanced면 손실에 앱별 가중치(class_weights)를 곱해 앱 간 개수 차이를 맞춘다.
    schedule이 cosine이면 학습률을 에포크마다 코사인 곡선으로 줄여 마지막 에포크 뒤 0이 되게 한다 (constant는 고정).
    history에 리스트를 넘기면 에포크마다 학습 손실·검증 점수를 덧붙인다 (metrics.json의 학습 곡선)."""
    loader = DataLoader(TensorDataset(torch.from_numpy(data.train.X), torch.from_numpy(data.train.y)),
                        batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    weight = torch.from_numpy(class_weights(data.train.y, len(data.classes))).to(device) if balanced else None
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if schedule == "cosine" else None
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
        record = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "loss": total / n}  # 이 에포크에 쓴 학습률
        if len(data.val.y):
            val = evaluate(model, data.val, len(data.classes), device)
            line += f"  val acc {val['accuracy']:.4f}  macro-F1 {val['macro_f1']:.4f}"
            record.update(val_accuracy=val["accuracy"], val_macro_f1=val["macro_f1"])
            if val["macro_f1"] > best_f1:
                best_f1, best_state = val["macro_f1"], copy.deepcopy(model.state_dict())
        print(line)
        if history is not None:
            history.append(record)
        if scheduler is not None:
            scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def validation_score(model: Model, data: Data, history: list[dict], device: str) -> dict | None:
    """결과에 남길 검증 점수 — 실험은 이 값으로 고른다 (시험 점수로 고르면 시험 세트에 맞춘 결과가 된다).
    신경망은 fit이 고른 에포크(검증 macro-F1 최고, 같으면 앞 에포크)의 값, 랜덤포레스트는 학습 뒤 검증 세트로 잰 값.
    검증 세트가 없으면 None."""
    scored = [r for r in history if "val_macro_f1" in r]
    if scored:
        best = max(scored, key=lambda r: r["val_macro_f1"])  # max는 같은 값이면 앞의 것 — fit의 > 와 같다
        return {"accuracy": best["val_accuracy"], "macro_f1": best["val_macro_f1"], "epoch": best["epoch"]}
    if isinstance(model, RandomForestClassifier) and len(data.val.y):
        val = evaluate(model, data.val, len(data.classes), device)
        return {"accuracy": val["accuracy"], "macro_f1": val["macro_f1"]}
    return None


def load_data(args: dict) -> Data:
    """학습 인자(argparse 값, 또는 저장한 metrics.json의 args)대로 데이터를 만든다 — 저장한 모델을 같은 조건으로
    다시 잴 때(src.ensemble)도 쓴다. 예전 학습의 args에 없는 인자는 기본값으로 본다."""
    if args["dataset"] == "cesnet":
        data = load_cesnet(size=args.get("cesnet_size", "XS"), train_size=args.get("train_size"))
    else:
        test_dates = args["test_dates"].split(",") if args.get("test_dates") else None
        data = load_self_collected(task=args.get("task", "app"), skip_first_s=args.get("skip_first", 0.0),
                                   proto=args.get("proto", "all"), test_dates=test_dates, seed=args.get("seed", 0))
    if args.get("skip_packets"):
        data = hide_handshake(data, args["skip_packets"])
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["cesnet", "self"], required=True)
    parser.add_argument("--model", choices=[*MODELS, "rf"], required=True, help="rf = 랜덤포레스트 기준선")
    parser.add_argument("--width", type=int, default=None,
                        help="신경망 폭: CNN은 첫 합성곱 채널 수, Transformer는 d_model (기본 64)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant",
                        help="학습률: constant(고정) 또는 cosine(에포크마다 코사인 곡선으로 줄여 마지막에 0)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cesnet-size", default="XS", choices=["XS", "S", "M", "L"])
    parser.add_argument("--train-size", type=int, default=None, help="CESNET 학습 샘플 수 제한 (빠른 실험용)")
    parser.add_argument("--task", choices=list(TASKS), default="app", help="직접 수집: app(앱 분류) 또는 screen(인스타 화면 분류)")
    parser.add_argument("--skip-first", type=float, default=0.0, help="직접 수집: 캡처 시작 뒤 이 초 안에 열린 플로우 제외")
    parser.add_argument("--proto", choices=list(PROTOS), default="all", help="직접 수집: all · quic · tcp")
    parser.add_argument("--test-dates", default=None, help="직접 수집: 시험용 날짜 (예: 1027,1028). 없으면 마지막 3일")
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced",
                        help="앱 간 개수 차이 맞추기: balanced(개수에 반비례하는 가중치) 또는 none")
    parser.add_argument("--skip-packets", type=int, default=0,
                        help="앞 N개 패킷(핸드셰이크)을 가리고 학습·평가 — 통신 라이브러리 지문 없이도 맞히는지 보는 실험")
    args = parser.parse_args()
    balanced = args.class_weight == "balanced"
    if not 0 <= args.skip_packets < N_PACKETS:
        parser.error(f"--skip-packets는 0 이상 {N_PACKETS} 미만이어야 한다")
    if args.width is not None and (args.model == "rf" or args.width < 1):
        parser.error("--width는 신경망(cnn · transformer)에만 1 이상으로 쓴다")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_data(vars(args))
    print(json.dumps(data.condition, ensure_ascii=False, indent=2))

    history: list[dict] = []
    started = time.perf_counter()
    if args.model == "rf":
        model = fit_forest(data, args.seed, balanced=balanced)
    else:
        model = fit(build_model(args.model, len(data.classes), args.width).to(args.device), data, args.epochs, args.batch_size, args.lr,
                    args.device, balanced=balanced, history=history, schedule=args.lr_schedule)
    train_seconds = time.perf_counter() - started
    val = validation_score(model, data, history, args.device)  # 처리량 시간 재기 전에 (랜덤포레스트는 여기서 예측한다)
    if val:
        picked = f"  (에포크 {val['epoch']})" if "epoch" in val else ""
        print(f"\n검증  정확도 {val['accuracy']:.4f}  macro-F1 {val['macro_f1']:.4f}{picked}")

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
        "val": val,  # 실험을 고를 때 쓰는 점수 (None = 검증 세트 없음)
        "condition": data.condition,
        "args": vars(args),
        "train_seconds": round(train_seconds, 1),
        # 모델만의 처리량 (tshark·플로우 묶기 제외). pcap 전체의 처리량은 python -m src.predict
        "infer_flows_per_s": round(len(data.test.y) / infer_seconds),
    }
    if history:
        metrics["history"] = history  # 에포크마다 학습 손실·검증 점수 (학습 곡선 그림용)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    per_class.to_csv(out / "per_class.csv")
    pd.DataFrame(test["confusion"], index=data.classes, columns=data.classes).to_csv(out / "confusion.csv")
    plot_confusion(out)
    if isinstance(model, nn.Module):
        # width는 다시 불러올 때 같은 모양으로 만들려고 남긴다 (None = 기본값)
        torch.save({"model": args.model, "width": args.width, "classes": data.classes, "state_dict": model.state_dict()},
                   out / "model.pt")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
