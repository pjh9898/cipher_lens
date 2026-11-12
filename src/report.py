"""runs/의 학습 결과를 결과 표(데이터셋 × 모델)로 모은다 — 보고서에 그대로 붙일 마크다운.

    python -m src.report          # 데이터셋·모델·조건마다 한 줄 — 시드 평균 ± 표준편차
    python -m src.report --all    # 모든 결과 (결과마다 한 줄, 시드 열과 함께)

정확도·macro-F1은 시험 점수, 검증 정확도·macro-F1은 실험을 비교하는 값이다 (experiments.md 1절) —
신경망은 끝 무렵 5 에포크 평균, 랜덤포레스트는 한 번 잰 값. 검증 점수를 남기기 전의 결과는 —.
학습 옵션이 기본값과 다르면 「조건」에 적는다. 조건 · 시드가 같으면 가장 최근 결과가 이전 결과를 덮고,
시드만 다른 결과는 한 줄로 묶어 시드 평균 ± 표준편차(%p)를 보인다.
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
SCORES = ["정확도", "macro-F1", "검증 정확도", "검증 macro-F1"]
SPREAD = " 표준편차"  # 시드를 평균한 점수의 표준편차 열 = 점수 열 이름 + SPREAD (to_markdown이 「평균 ± 표준편차」로 합친다)


def dataset_name(condition: dict) -> str:
    if condition.get("dataset") == "self_collected":
        return f"직접 수집 {condition['task']}"
    return "CESNET"


def describe_split(condition: dict) -> str:
    if "test_period" in condition:
        return f"{condition['train_period']} → {condition['test_period']}"
    return f"{','.join(condition['train_dates'])} → {','.join(condition['test_dates'])}"


def validation_scores(metrics: dict, k: int = 5) -> tuple[float | None, float | None]:
    """실험을 비교하는 검증 (정확도, macro-F1). 신경망은 끝 무렵 k 에포크의 평균이다 — 가장 좋은 에포크의 점수는
    곡선이 오르내릴수록 운으로 부풀어서 비교에 쓰지 않는다. 학습 곡선이 없으면(랜덤포레스트) 한 번 잰 값."""
    scored = [r for r in metrics.get("history", []) if "val_macro_f1" in r]
    if scored:
        tail = scored[-k:]
        return float(np.mean([r["val_accuracy"] for r in tail])), float(np.mean([r["val_macro_f1"] for r in tail]))
    val = metrics.get("val") or {}
    return val.get("accuracy"), val.get("macro_f1")


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
        if c.get("skip_packets"):
            extra.append(f"앞 패킷 {c['skip_packets']}개 가림")
        if c.get("val_days"):
            extra.append(f"검증 마지막 {c['val_days']}일")
        if m["args"].get("class_weight") == "none":
            extra.append("클래스 가중치 없음")
        if m["args"].get("lr_schedule", "constant") == "cosine":
            extra.append("코사인 학습률")
        if m["args"].get("width"):
            extra.append(f"폭 {m['args']['width']}")
        val_accuracy, val_macro_f1 = validation_scores(m)
        rows.append({
            "데이터셋": dataset_name(c),
            "모델": m["args"]["model"],
            "정확도": m["accuracy"],
            "macro-F1": m["macro_f1"],
            "검증 정확도": val_accuracy,
            "검증 macro-F1": val_macro_f1,
            "앱 수": c["classes"],
            "분할 (학습 → 시험)": describe_split(c),
            "플로우 (학습/검증/시험)": f"{flows['train']}/{flows['val']}/{flows['test']}",
            "조건": " · ".join(extra),
            "시드": m["args"].get("seed", 0),
            "run": path.parent.name,
        })
    return pd.DataFrame(rows)


def average_seeds(runs: pd.DataFrame) -> pd.DataFrame:
    """조건이 같고 시드만 다른 결과를 한 줄로 묶는다 — 점수는 시드 평균, 시드 간 표준편차는 「… 표준편차」 열(시드 1개면 비움).
    검증 점수가 없는 결과(검증 점수를 남기기 전 학습)는 검증 평균에서 빠진다. 앱 수 · 분할 · 플로우는 가장 최근 결과의 값."""
    rows = []
    for _, group in runs.sort_values("run").groupby(["데이터셋", "모델", "조건"], sort=False):
        row = group.iloc[-1].drop("시드").to_dict()
        for col in SCORES:
            values = pd.to_numeric(group[col], errors="coerce")  # None → NaN
            row[col] = values.mean()
            row[col + SPREAD] = values.std() if values.count() > 1 else np.nan
        row["시드 수"] = len(group)
        if len(group) > 1:
            row["run"] = f"{row['run']} 외 {len(group) - 1}개"
        rows.append(row)
    columns = [c for c in runs.columns if c not in ("시드", "run")] + ["시드 수", "run"]
    return pd.DataFrame(rows)[columns + [col + SPREAD for col in SCORES]]


def rank(values: pd.Series, order: list[str]) -> pd.Series:
    return values.map({v: i for i, v in enumerate(order)}).fillna(len(order))


def results_table(runs: pd.DataFrame, latest_only: bool = True) -> pd.DataFrame:
    """CESNET → 앱 분류 → 화면 분류, 그 안에서 CNN → Transformer → 랜덤포레스트(기준선) 순.
    latest_only면 조건 · 시드마다 가장 최근 결과만 남기고, 시드만 다른 결과는 한 줄로 묶는다 (average_seeds)."""
    if runs.empty:
        return runs
    if latest_only:  # run 이름이 시각으로 시작하므로 마지막이 가장 최근. 조건(프로토콜·앞 패킷 가림 등)이 다르면 따로 남긴다
        runs = runs.sort_values("run").groupby(["데이터셋", "모델", "조건", "시드"]).tail(1)
        runs = average_seeds(runs)
    runs = runs.assign(_d=rank(runs["데이터셋"], DATASET_ORDER), _m=rank(runs["모델"], MODEL_ORDER))
    return runs.sort_values(["_d", "_m", "run"]).drop(columns=["_d", "_m"]).reset_index(drop=True)


def to_markdown(df: pd.DataFrame) -> str:
    """마크다운 표. 비율은 %로 보이고, 「… 표준편차」 열이 있으면 그 점수 칸에 합쳐 「평균 ± 표준편차(%p)」로 보인다."""
    def is_number(v) -> bool:
        return isinstance(v, float) and not np.isnan(v)

    def cell(row: dict, col: str) -> str:
        v, spread = row[col], row.get(col + SPREAD)
        if is_number(v) and is_number(spread):
            return f"{v:.2%} ± {100 * spread:.2f}"
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v:.2%}" if isinstance(v, float) else str(v)

    columns = [c for c in df.columns if not c.endswith(SPREAD)]
    lines = ["| " + " | ".join(columns) + " |", "|" + " --- |" * len(columns)]
    lines += ["| " + " | ".join(cell(row, c) for c in columns) + " |" for row in df.to_dict("records")]
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
    parser.add_argument("--all", action="store_true", help="시드를 묶지 않고, 같은 조건의 이전 결과도 모두 보인다")
    args = parser.parse_args()

    table = results_table(load_runs(), latest_only=not args.all)
    if table.empty:
        raise SystemExit(f"{RUNS_DIR}에 학습 결과가 없다. 먼저 python -m src.train으로 학습한다.")
    print(to_markdown(table))


if __name__ == "__main__":
    main()
