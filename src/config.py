"""경로·상수를 한 곳에 둔다."""
import os
from pathlib import Path

# 데이터는 저장소 밖에 둔다. 다른 PC에서는 환경변수 CIPHERLENS_DATA로 바꾼다.
DATA_ROOT = Path(os.environ.get("CIPHERLENS_DATA", "D:/datasets"))
CESNET_DIR = DATA_ROOT / "CESNET-QUIC22"
SELF_COLLECTED_DIR = DATA_ROOT / "self_collected"
PROCESSED_DIR = DATA_ROOT / "processed"

# 학습 결과 (모델 가중치·지표) — .gitignore
RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"

# PPI: 플로우마다 앞 30개 패킷. 행 순서는 cesnet_datazoo.constants
# (IPT_POS=0, DIR_POS=1, SIZE_POS=2)와 같다.
N_PACKETS = 30
IPT, DIR, SIZE = 0, 1, 2
