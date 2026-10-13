"""경로·상수를 한 곳에 둔다."""
import os
from pathlib import Path

# 데이터는 저장소 밖에 둔다. 다른 PC에서는 환경변수 CIPHERLENS_DATA로 바꾼다.
DATA_ROOT = Path(os.environ.get("CIPHERLENS_DATA", "D:/datasets"))
CESNET_DIR = DATA_ROOT / "CESNET-QUIC22"
SELF_COLLECTED_DIR = DATA_ROOT / "self_collected"
PROCESSED_DIR = DATA_ROOT / "processed"

# 분류 과제별 라벨 = 직접 수집 파일 이름의 라벨. 여기 없는 라벨의 파일은 이름 오타로 보고 건너뛴다.
# 두 과제는 수집 방법이 달라서 한 모델에 섞지 않는다.
TASKS = {
    # 앱 분류: 앱을 완전히 끈 상태에서 캡처를 켜고 앱을 연다
    "app": ("youtube", "webtoon", "instagram"),
    # 인스타 화면 분류: 인스타를 켜고 그 화면에 들어간 뒤 캡처를 켠다
    "screen": ("instagram-reels", "instagram-feed", "instagram-explore"),
}
LABELS = tuple(label for labels in TASKS.values() for label in labels)

# 학습 결과 (모델 가중치·지표) — .gitignore
RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"

# PPI: 플로우마다 앞 30개 패킷. 행 순서는 cesnet_datazoo.constants
# (IPT_POS=0, DIR_POS=1, SIZE_POS=2)와 같다.
N_PACKETS = 30
IPT, DIR, SIZE = 0, 1, 2
