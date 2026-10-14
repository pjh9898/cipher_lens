"""학습한 모델로 pcap 분류 — 프로토타입(app.py)과 처리량·저장량 측정이 같이 쓴다.

플로우는 모델을 학습할 때의 조건(metrics.json의 condition — proto, skip_first_s)대로 거른다.
학습과 시연이 같은 입력을 보게 하려고.

    python -m src.predict D:/datasets/self_collected/youtube_1027_1.pcap        # runs/의 가장 최근 모델
    python -m src.predict --run runs/20261101-120000_self-app_cnn "D:/datasets/self_collected/*_1027_*.pcap"
    python -m src.predict D:/datasets/self_collected                            # 폴더 안 pcap 전부
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.config import N_PACKETS, PROTOS, RUNS_DIR
from src.models import build_model
from src.preprocess import flows_from_packets, packets_from_tshark, parse_pcap_name, run_tshark

FEATURE_BYTES_PER_FLOW = 3 * N_PACKETS * 4  # float32 (3, 30) — 플로우 하나를 저장하는 데 드는 크기


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: str, batch_size: int = 4096) -> np.ndarray:
    """원시 PPI (N, 3, 30) → 앱별 확률 (N, 앱 수)."""
    if len(X) == 0:  # Transformer는 빈 배치를 못 받는다 → 한 줄을 넣어 앱 수만 알아낸다
        return predict_proba(model, np.zeros((1, 3, N_PACKETS), dtype=np.float32), device)[:0]
    model.eval()
    return np.concatenate([
        torch.softmax(model(torch.from_numpy(X[i:i + batch_size]).to(device)), dim=1).cpu().numpy()
        for i in range(0, len(X), batch_size)
    ])


@dataclass
class Classifier:
    model: nn.Module
    classes: list[str]
    metrics: dict  # 학습 때 남긴 metrics.json (정확도·macro-F1·condition·args)
    device: str

    @property
    def condition(self) -> dict:
        return self.metrics.get("condition", {})


def list_runs(root: Path = RUNS_DIR) -> list[Path]:
    """model.pt가 있는 학습 결과 폴더, 최근 것부터 (폴더 이름이 시각으로 시작한다)."""
    return sorted((p.parent for p in Path(root).glob("*/model.pt")), reverse=True)


def load_classifier(run: Path, device: str | None = None) -> Classifier:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(Path(run) / "model.pt", map_location=device)
    model = build_model(ckpt["model"], len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    predict_proba(model, np.zeros((0, 3, N_PACKETS), dtype=np.float32), device)  # GPU 준비 시간을 처리량에서 빼려고 미리 한 번
    metrics_path = Path(run) / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    return Classifier(model, list(ckpt["classes"]), metrics, device)


def keep_like_training(meta: pd.DataFrame, condition: dict) -> np.ndarray:
    """학습 때와 같은 조건(프로토콜, 캡처 직후 몇 초 제외)으로 남길 플로우."""
    protos = PROTOS[condition.get("proto", "all")]
    return (meta["proto"].isin(protos) & (meta["offset"] >= condition.get("skip_first_s", 0.0))).to_numpy()


@dataclass
class PcapResult:
    name: str
    flows: pd.DataFrame  # 플로우마다 offset · proto · ppi_len · app(예측) · confidence
    seconds: dict[str, float]  # 단계별 처리 시간 — tshark · flows(플로우 묶기) · model
    pcap_bytes: int
    true_label: str | None  # 파일 이름 규칙에 맞으면 그 라벨
    accuracy: float | None  # 파일 이름의 라벨을 정답으로 본 정확도. 그 라벨이 모델의 앱 목록에 없으면 None

    @property
    def total_seconds(self) -> float:
        return sum(self.seconds.values())

    @property
    def flows_per_s(self) -> float:
        return len(self.flows) / self.total_seconds if self.total_seconds else 0.0

    @property
    def feature_bytes(self) -> int:
        return len(self.flows) * FEATURE_BYTES_PER_FLOW


def classify_pcap(clf: Classifier, pcap: Path) -> PcapResult:
    pcap = Path(pcap)
    t0 = time.perf_counter()
    packets = packets_from_tshark(run_tshark(pcap))
    t1 = time.perf_counter()
    ppi, meta = flows_from_packets(packets)
    keep = keep_like_training(meta, clf.condition)
    ppi, meta = ppi[keep], meta[keep].reset_index(drop=True)
    t2 = time.perf_counter()
    proba = predict_proba(clf.model, ppi, clf.device)
    t3 = time.perf_counter()

    flows = meta[["offset", "proto", "ppi_len"]].assign(
        app=np.array(clf.classes)[proba.argmax(1)],
        confidence=proba.max(1),
    )
    parsed = parse_pcap_name(pcap.name)
    true_label = parsed["label"] if parsed else None
    accuracy = float((flows["app"] == true_label).mean()) if true_label in clf.classes and len(flows) else None
    return PcapResult(
        name=pcap.name, flows=flows, seconds={"tshark": t1 - t0, "flows": t2 - t1, "model": t3 - t2},
        pcap_bytes=pcap.stat().st_size, true_label=true_label, accuracy=accuracy,
    )


def summarize(flows: pd.DataFrame) -> pd.DataFrame:
    """앱별 플로우 수 · 비율 · 평균 확신도, 플로우 많은 순."""
    by_app = flows.groupby("app")
    out = pd.DataFrame({"플로우": by_app.size(), "평균 확신도": by_app["confidence"].mean()})
    out.insert(1, "비율", out["플로우"] / out["플로우"].sum())
    return out.sort_values("플로우", ascending=False)


def expand_paths(paths: list[str]) -> list[Path]:
    """PowerShell은 *를 풀어 주지 않으므로 직접 푼다. 폴더를 주면 안의 pcap 전부."""
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(q for q in p.iterdir() if q.suffix in (".pcap", ".pcapng"))
        elif any(ch in p.name for ch in "*?["):
            out += sorted(p.parent.glob(p.name))
        else:
            out.append(p)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pcaps", nargs="+", help="pcap 파일·폴더·패턴")
    parser.add_argument("--run", type=Path, default=None, help="학습 결과 폴더 (기본: runs/의 가장 최근)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    runs = list_runs()
    run = args.run or (runs[0] if runs else None)
    if run is None:
        raise SystemExit("runs/에 학습한 모델이 없다. 먼저 python -m src.train으로 학습한다.")
    pcaps = expand_paths(args.pcaps)
    if not pcaps:
        raise SystemExit(f"pcap이 없다: {' '.join(args.pcaps)}")
    clf = load_classifier(run, args.device)
    print(f"모델: {run.name} ({clf.device})")
    print(json.dumps(clf.condition, ensure_ascii=False))

    results = []
    for pcap in pcaps:
        r = classify_pcap(clf, pcap)
        results.append(r)
        line = f"\n{r.name}: 플로우 {len(r.flows)}개, {r.total_seconds:.2f}초 ({r.flows_per_s:.0f} 플로우/초)"
        if r.accuracy is not None:
            line += f", 정답 {r.true_label} → 정확도 {r.accuracy:.1%}"
        print(line)
        for app, row in summarize(r.flows).head(5).iterrows():
            print(f"  {app:<24} {row['플로우']:>5.0f}  {row['비율']:6.1%}  확신도 {row['평균 확신도']:.2f}")

    n_flows = sum(len(r.flows) for r in results)
    stage = {k: sum(r.seconds[k] for r in results) for k in ("tshark", "flows", "model")}
    total_s = sum(stage.values())
    pcap_mb = sum(r.pcap_bytes for r in results) / 1e6
    feature_kb = n_flows * FEATURE_BYTES_PER_FLOW / 1e3
    print(f"\n합계: pcap {len(results)}개, 플로우 {n_flows}개, {total_s:.2f}초 → {n_flows / total_s if total_s else 0:.0f} 플로우/초")
    print(f"단계별: tshark {stage['tshark']:.2f}초 · 플로우 묶기 {stage['flows']:.2f}초 · 모델 {stage['model']:.2f}초")
    print(f"저장량: pcap {pcap_mb:.1f}MB → 특징값 {feature_kb:.1f}KB (플로우당 {FEATURE_BYTES_PER_FLOW}바이트, "
          f"pcap의 {feature_kb / 1e3 / pcap_mb if pcap_mb else 0:.2%})")


if __name__ == "__main__":
    main()
