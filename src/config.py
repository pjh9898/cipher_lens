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

# 앱별 도메인. 직접 수집 파일에 다른 앱의 연결이 섞이면(PCAPdroid 대상 앱 설정 실수 등) 전처리에서 SNI로 골라 뺀다.
# SNI는 거르는 데만 쓰고 모델 입력에는 넣지 않는다 (CESNET-QUIC22도 SNI로 라벨을 붙였다).
# 여러 앱이 같이 쓰는 광고·분석 도메인(doubleclick, firebase 등)은 넣지 않는다 — 라벨 노이즈로 남긴다.
# 화면 분류 라벨은 '-' 앞이 앱 이름이다 (instagram-reels → instagram).
APP_DOMAINS = {
    "youtube": ("youtube.com", "googlevideo.com", "ytimg.com", "ggpht.com", "youtubei.googleapis.com"),
    "webtoon": ("naver.com", "pstatic.net", "navercorp.com"),
    "instagram": ("instagram.com", "cdninstagram.com", "facebook.com", "fbcdn.net"),
}

# 학습에 쓸 프로토콜 (train --proto). 전처리는 443 포트만 남기므로 udp = QUIC, tcp = TLS.
PROTOS = {"all": ("tcp", "udp"), "quic": ("udp",), "tcp": ("tcp",)}

# 학습 결과 (모델 가중치·지표) — .gitignore
RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"

# PPI: 플로우마다 앞 30개 패킷. 행 순서는 cesnet_datazoo.constants
# (IPT_POS=0, DIR_POS=1, SIZE_POS=2)와 같다.
N_PACKETS = 30
IPT, DIR, SIZE = 0, 1, 2
