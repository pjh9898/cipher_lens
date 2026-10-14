# CLAUDE.md

CipherLens — 캡스톤디자인2(기업연계형, 2026-2학기) 1인 과제.
플로우 하나의 **앞 30개 패킷의 (크기, 방향, 간격)** 만 보고 어떤 앱의 트래픽인지 맞히는 딥러닝 모델(1D-CNN vs 작은 Transformer)과, pcap을 넣으면 앱별 분류 결과를 보여 주는 Streamlit 프로토타입을 만든다.

- 계획·배경·데이터·참고 자료의 정본: [README.md](README.md)
- 할 일·일정: [todo.md](todo.md) — 작업을 마치면 해당 체크박스를 갱신한다.
- 현재 상태: 전처리·데이터·모델·학습·추론·결과 표·프로토타입(`app.py`) 코드와 가짜 데이터 테스트까지 있음. 실제 데이터(CESNET 다운로드, 직접 수집 pcap)로는 아직 돌려 보지 않았다. 전처리는 가짜 pcap + 진짜 tshark 4.6.9로 확인함(`tests/test_tshark.py`).

## 명령어

Windows · PowerShell. 가상환경은 `.venv` (Python 3.12, torch CUDA 12.6).

```powershell
.\.venv\Scripts\python.exe -m pytest                       # 코드를 바꾸면 돌린다
.\.venv\Scripts\python.exe -m src.preprocess --status      # 직접 수집 현황
.\.venv\Scripts\python.exe -m src.preprocess               # pcap → D:/datasets/processed/self_collected.npz (tshark 필요)
.\.venv\Scripts\python.exe -m src.train --dataset cesnet|self --model cnn|transformer|rf [--task app|screen] [--skip-first 초] [--proto all|quic|tcp]
.\.venv\Scripts\python.exe -m src.report                   # runs/ → 결과 표 (마크다운)
.\.venv\Scripts\python.exe -m src.predict <pcap·폴더·패턴>  # 분류 + 처리량·저장량 (runs/의 최근 모델)
.\.venv\Scripts\streamlit.exe run app.py                   # 프로토타입
```

`src`는 패키지로 불러오므로 저장소 루트에서 `-m`으로 실행한다.

## 바꾸면 안 되는 것

- 정식 과제명 「응용레벨 암호화 트래픽 AI기반 분석 기술 개발」은 수정하지 않는다 (제출한 과제수행계획서의 이름). 부제·짧은 이름은 README 1절 표기 그대로 쓴다.
- 회사(하잎아이피) 데이터는 쓰지 않는다. 데이터는 CESNET-QUIC22(주) + 직접 수집(PCAPdroid)만. ISCX VPN-nonVPN 2016은 선택.
- 고속 처리·대용량 저장은 만들지 않는다 — 처리량(초당 플로우 수)·저장량 측정으로 갈음한다.

## 분류 대상 (직접 수집)

라벨은 [src/config.py](src/config.py)의 `TASKS` 한 곳에만 둔다. 과제 두 개는 **한 모델에 섞지 않는다.**

| 과제 (`--task`) | 라벨 | 수집 방법 |
| --- | --- | --- |
| `app` 앱 분류 (본 과제, 계획서의 응용 분류) | `youtube` `webtoon` `instagram` | 앱을 강제 종료한 상태에서 캡처 시작 → 앱 열기 |
| `screen` 인스타 화면 분류 (추가 분석) | `instagram-reels` `instagram-feed` `instagram-explore` | 인스타를 켜고 그 화면에 들어간 뒤 캡처 시작 |

- 섞으면 안 되는 이유: `instagram`에도 피드·릴스가 들어 있어 라벨이 겹치고, 캡처를 켠 시점이 달라 모델이 앱 대신 수집 방법을 배울 수 있다.
- `screen` 파일은 캡처 전에 열린 연결이 많다 → 중간부터 잡힌 플로우는 전처리에서 빼고, 캡처 직후 재연결은 `--skip-first`로 뺀다 (값은 첫날 데이터의 `offset` 분포를 보고 정한다).

## 모델 입력 규칙

- 플로우 = 5-튜플(출발지·목적지 IP·포트, 프로토콜)로 묶은 패킷.
- 플로우마다 앞 30개 패킷의 크기(페이로드 바이트), 방향(+1 = 클라이언트→서버, −1 = 서버→클라이언트), 간격(앞 패킷과의 도착 시간 차).
- 모양은 CESNET-QUIC22의 PPI와 같게 (N, 3, 30), 행 순서 `[IPT(ms), DIR, SIZE]` (`cesnet_datazoo.constants`의 `IPT_POS=0, DIR_POS=1, SIZE_POS=2`). 상수는 [src/config.py](src/config.py)에만 둔다.
- 모델에는 원시 PPI를 넣는다. 정규화는 모델 안(`models.Normalize`)에 있으므로 학습·평가·프로토타입이 같은 입력을 쓴다.
- TCP는 CESNET처럼 페이로드 0인 패킷과 재전송을 빼고, QUIC은 모두 넣는다.
- 캡처 전에 열려 중간부터 잡힌 플로우는 뺀다: TCP는 SYN, QUIC은 long header(`quic.header_form`)로 시작해야 한다.
- **IP·포트·SNI·페이로드 바이트는 입력에 넣지 않는다** (모델이 답을 외우는 지름길이 된다). 전처리 코드를 바꿀 때마다 다시 확인한다.

## 평가 규칙

- 학습/시험은 **수집 날짜(또는 세션) 단위**로 나눈다. 무작위 분할 금지. CESNET은 학습 `W-2022-44` / 시험 `W-2022-45`.
- 지표는 정확도 + macro-F1을 함께 적고, 혼동 행렬을 남긴다.
- 85% 목표는 측정 조건(데이터셋·앱 수·분할 방법)과 함께만 보고한다.
- 결과 표는 데이터셋(CESNET / 직접 수집 `app` / 직접 수집 `screen`)별 × 모델별(CNN vs Transformer).
- 프로토타입 시연 입력은 직접 수집 pcap 중 학습에 쓰지 않은 날짜분.

## 코드 구성

```
notebooks/          탐색·그래프 (.ipynb) — 01 CESNET, 02 직접 수집(--skip-first·--proto 정하기)
src/config.py       데이터 경로(CIPHERLENS_DATA, 기본 D:/datasets)·PPI 상수·과제별 라벨(TASKS)·프로토콜 선택(PROTOS)
src/preprocess.py   pcap → tshark CSV → 플로우 → 앞 30개 패킷 행렬, 파일 이름 규칙
src/data.py         CESNET·직접 수집 불러오기 → Data(train/val/test, classes, condition)
src/models.py       1D-CNN, 작은 Transformer
src/train.py        학습·평가 → runs/<시각>_<데이터>_<모델>/ (metrics.json · per_class.csv · confusion.csv/png · model.pt). rf(랜덤포레스트 기준선)는 model.pt 없음
src/predict.py      runs/의 모델로 pcap 분류 (학습 condition대로 플로우를 거름), 처리량·저장량
src/report.py       runs/ → 결과 표, 혼동 행렬 그림
tests/              가짜 데이터 테스트 (실제 데이터 불필요). test_tshark.py만 tshark가 있을 때 돈다
scripts/            pull_pcaps.ps1 (adb로 폰 → PC)
app.py              Streamlit 프로토타입 — 화면만, 계산은 src/predict.py
```

- `model.pt`는 `torch.load` 기본값(`weights_only=True`)으로 다시 불러온다 → 저장하는 값에 numpy 타입(`np.str_` 등)을 넣지 않는다.

- 결과를 보고할 때는 `Data.condition`(데이터셋·앱 수·분할 방법·플로우 수)을 함께 적는다.

- 다시 쓰는 코드(전처리·모델·학습)는 `.py`, 탐색·그래프는 노트북.
- 프로토타입 화면은 최소로 — 시간은 모델에 쓴다.

## 데이터·보안

- 데이터는 저장소 밖 `D:/datasets/`(`CESNET-QUIC22/`, `self_collected/`)에 둔다. 저장소 안에 데이터를 만들거나 복사하지 않는다.
- 바탕 화면은 OneDrive 동기화 폴더(`C:\Users\jhpark\OneDrive\바탕 화면`)라 pcap을 두지 않는다. 바탕 화면의 `pcap.lnk`는 `D:\datasets\self_collected` 바로가기다.
- `*.pcap`, `*.pcapng`, `data/`는 커밋하지 않는다 (`.gitignore`).
- pcap 원본은 공개하지 않는다. 공유가 필요하면 추출한 특징값만.
- 직접 수집 파일명은 `라벨_날짜_순번.pcap` (예: `youtube_1013_1.pcap`, `instagram-reels_1013_1.pcap`) — 파일명이 곧 라벨이고 날짜 분할의 기준이다.
- 본인 기기 트래픽만 수집한다 (통신비밀보호법).

## 환경

- Windows · VS Code(Python·Jupyter 확장) · Python 3.10 이상 (cesnet-datazoo 요구).
- 패키지: `cesnet-datazoo torch scikit-learn pandas matplotlib streamlit`
- pcap → CSV는 tshark (Wireshark 설치 시 포함). 지금 PC: Wireshark 4.6.9, `C:\Program Files\Wireshark\tshark.exe` (PATH에는 없음 — `preprocess.find_tshark`가 이 경로를 기본으로 찾는다). Npcap은 없음 — 파일만 읽으므로 필요 없다.
- GPU가 없으면 CPU + CESNET XS로 시작하고, 느리면 학습만 Google Colab으로.

## 문서 작성

- README·todo·보고서는 한국어로, README처럼 쉬운 말로 쓴다.
- 한계(라벨 노이즈 — 광고·분석 서버 연결이 섞임, 인스타 화면 라벨에 선로딩 트래픽이 섞임, PCAPdroid VPN 방식의 크기·타이밍 차이, 데이터 규모)는 숨기지 않고 한계로 적는다.
