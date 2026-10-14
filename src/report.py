"""runs/의 학습 결과를 결과 표(데이터셋 × 모델)로 모은다 — 보고서에 그대로 붙일 마크다운.

    python -m src.report          # 데이터셋·모델마다 가장 최근 결과
    python -m src.report --all    # 모든 결과
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.config import RUNS_DIR

DATASET_ORDER = ["CESNET", "직접 수집 app", "직접 수집 screen"]
MODEL_ORDER = ["cnn", "transformer", "rf"]  # rf = 기준선


def dataset_name(condition: dict) -> str:
    if condition.get("dataset") == "self_collected":
        return f"직접 수집 {condition['task']}"
    return "CESNET"


def describe_split(condition: dict) -> str:
    if "test_period" in condition:
        return f"{condition['train_period']} → {condition['test_period']}"
    return f"{','.join(condition['train_dates'])} → {','.join(condition['test_dates'])}"


def load_runs(root: Path = RUNS_DIR) -> pd.DataFrame:
    """학습 결과 폴더마다 한 줄 — 85% 같은 숫자를 측정 조건과 함께 보이도록."""
    rows = []
    for path in sorted(Path(root).glob("*/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        c = m["condition"]
        flows = c["flows"]
        extra = []
        if c.get("skip_first_s"):
            extra.append(f"앞 {c['skip_first_s']:g}초 제외")
        if c.get("proto", "all") != "all":
            extra.append(c["proto"])
        rows.append({
            "데이터셋": dataset_name(c),
            "모델": m["args"]["model"],
            "정확도": m["accuracy"],
            "macro-F1": m["macro_f1"],
            "앱 수": c["classes"],
            "분할 (학습 → 시험)": describe_split(c),
            "플로우 (학습/검증/시험)": f"{flows['train']}/{flows['val']}/{flows['test']}",
            "조건": " · ".join(extra),
            "run": path.parent.name,
        })
    return pd.DataFrame(rows)


def rank(values: pd.Series, order: list[str]) -> pd.Series:
    return values.map({v: i for i, v in enumerate(order)}).fillna(len(order))


def results_table(runs: pd.DataFrame, latest_only: bool = True) -> pd.DataFrame:
    """CESNET → 앱 분류 → 화면 분류, 그 안에서 CNN → Transformer → 랜덤포레스트(기준선) 순."""
    if runs.empty:
        return runs
    if latest_only:  # run 이름이 시각으로 시작하므로 마지막이 가장 최근
        runs = runs.sort_values("run").groupby(["데이터셋", "모델"]).tail(1)
    runs = runs.assign(_d=rank(runs["데이터셋"], DATASET_ORDER), _m=rank(runs["모델"], MODEL_ORDER))
    return runs.sort_values(["_d", "_m", "run"]).drop(columns=["_d", "_m"]).reset_index(drop=True)


def to_markdown(df: pd.DataFrame) -> str:
    def cell(v) -> str:
        return f"{v:.2%}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + " --- |" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def plot_confusion(run: Path) -> Path:
    """confusion.csv → confusion.png (행 = 정답, 행마다 비율). 앱이 많으면(CESNET) 숫자는 생략한다."""
    cm = pd.read_csv(Path(run) / "confusion.csv", index_col=0)
    ratio = cm.to_numpy() / np.maximum(cm.to_numpy().sum(1, keepdims=True), 1)
    n = len(cm)
    with matplotlib.rc_context({"font.family": "Malgun Gothic", "axes.unicode_minus": False}):
        fig = Figure(figsize=(max(5, 0.3 * n + 2), max(4, 0.3 * n + 1.5)))
        ax = fig.subplots()
        image = ax.imshow(ratio, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n), cm.columns, rotation=90 if n > 10 else 45, ha="right" if n <= 10 else "center")
        ax.set_yticks(range(n), cm.index)
        ax.set_xlabel("예측")
        ax.set_ylabel("정답")
        if n <= 15:
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f"{ratio[i, j]:.2f}\n({cm.iat[i, j]})", ha="center", va="center", fontsize=8,
                            color="white" if ratio[i, j] > 0.5 else "black")
        fig.colorbar(image, ax=ax, fraction=0.046)
        ax.set_title(f"혼동 행렬 — {Path(run).name}", fontsize=9)
        out = Path(run) / "confusion.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="같은 데이터셋·모델의 이전 결과도 모두 보인다")
    args = parser.parse_args()

    table = results_table(load_runs(), latest_only=not args.all)
    if table.empty:
        raise SystemExit(f"{RUNS_DIR}에 학습 결과가 없다. 먼저 python -m src.train으로 학습한다.")
    print(to_markdown(table))


if __name__ == "__main__":
    main()
