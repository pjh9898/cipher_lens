# CipherLens

> **응용레벨 암호화 트래픽 AI기반 분석 기술 개발**
> 복호화 없이 패킷 패턴으로 TLS 1.3·QUIC 응용을 분류하는 딥러닝 모델

할 일 목록은 [todo.md](./todo.md).

---

## 1. 과제 정보

| 항목 | 내용 |
| --- | --- |
| 과목 | 캡스톤디자인2 (기업연계형) · 2026-2학기 |
| 지도교수 | 김명섭 교수님 |
| 참여기업 | (주)하잎아이피 |
| 구성 | 1인 (팀장) |
| 정식 과제명 | 응용레벨 암호화 트래픽 AI기반 분석 기술 개발 — 2026-09-21 제출한 과제수행계획서의 이름. **바꾸지 않는다** (바꾸면 계획서를 다시 낼 수 있음) |
| 부제 | 복호화 없이 패킷 패턴으로 TLS 1.3·QUIC 응용을 분류하는 딥러닝 모델 (발표·포스터·보고서용) |
| 짧은 이름 | CipherLens (저장소·시연 화면용) |
| 기업 해시태그 | #응용 트래픽 분류 #암호화 트래픽 분석 #실시간 분석 #딥러닝 분석 #대용량 트래픽 저장 #고속 네트워크 처리 |

### 계획서에 적은 내용 (요약)

- **목적**: TLS 1.3·QUIC 확산으로 DPI 방식의 응용 식별이 어려워졌다. 복호화 없이 트래픽 패턴만으로 응용을 분류하는 AI 기술을 만든다.
- **방법**: 패킷 길이 시퀀스·도착 간격·플로우 통계를 뽑고 CNN·Transformer 모델로 분류한다. 공개 데이터셋과 자체 수집 트래픽으로 학습·검증하고, 결과를 볼 수 있는 프로토타입을 만든다.
- **기업 역할(계획서 기준)**: 실환경 트래픽 데이터와 고속 처리·저장 환경 제공, 요구사항 정의·실무 검증.
- **목표**: 공개 및 자체 수집 데이터 기준 분류 정확도 85% 이상, 분석 프로토타입.
- **추진 단계**: ① 관련 연구 조사·공개 데이터셋 확보 → ② 전처리·특징 추출 모듈 → ③ 딥러닝 모델 학습·성능 비교 → ④ 프로토타입 통합·검증·결과 정리.

### 계획서와 달라진 점

- **회사 데이터는 쓰지 않는다.** 공개 데이터 + 내가 직접 수집한 데이터만 쓴다.
  → 지도교수님께 확인받고, 하잎아이피에도 알린다 (todo).
- 고속 처리·대용량 저장은 **직접 만들지 않는다.** 회사 처리·저장 환경을 쓸지는 미정 — 쓰지 않으면 프로토타입의 처리량(초당 플로우 수)·저장량을 재서 보고하는 것으로 갈음한다.

---

## 2. 무엇을 만드나

**플로우(연결) 하나의 앞 30개 패킷의 (크기, 방향, 간격)만 보고 어떤 앱의 트래픽인지 맞히는 모델**과, pcap을 넣으면 앱별 분류 결과를 보여 주는 프로토타입.

| 해시태그 | 이 과제에서 |
| --- | --- |
| 응용 트래픽 분류 · 암호화 트래픽 분석 · 딥러닝 분석 | 모델 (CNN vs Transformer) |
| 실시간 분석 | 프로토타입 — pcap 입력 → 분류 결과 화면 |
| 고속 네트워크 처리 · 대용량 트래픽 저장 | 만들지 않음 — 처리량·저장량 측정으로 갈음 (회사 환경 사용 여부 미정) |

### 분류 대상 (직접 수집)

과제 두 개를 따로 학습·평가한다. 수집 방법이 달라서 한 모델에 섞지 않는다.

| 과제 | 라벨 (파일 이름) | 수집 방법 |
| --- | --- | --- |
| **앱 분류** (본 과제) | `youtube` 유튜브 · `webtoon` 네이버웹툰 · `instagram` 인스타그램 | 앱을 **강제 종료**한 상태에서 캡처 시작 → 앱 열기 → 평소처럼 사용 (인스타는 피드·릴스) |
| **인스타 화면 분류** (추가 분석) | `instagram-reels` 릴스 · `instagram-feed` 피드 · `instagram-explore` 탐색 탭 | 인스타를 켜고 그 화면에 들어간 뒤 캡처 시작 → 그 화면에서만 사용 (피드는 켜자마자 캡처) |

- 앱 분류의 「강제 종료」는 설정 → 애플리케이션 → 그 앱 → 강제 종료. 최근 앱 목록에서 밀어 닫기만 하면 앱이 백그라운드에 남아 있을 수 있다. 첫 수집 파일(`instagram_1014_1`)에서 캡처 전에 열린 QUIC 연결이 트래픽의 85%였고, 그 연결은 전처리에서 빠진다. `python -m src.preprocess`가 이런 파일에 경고를 띄운다.
- 두 과제를 섞으면 안 되는 이유: `instagram`에도 피드·릴스 트래픽이 들어 있어 라벨이 겹치고, 화면 라벨은 캡처를 켠 시점이 달라서 모델이 앱 대신 「캡처를 언제 켰는지」를 배울 수 있다.
- 라벨 목록은 `src/config.py`의 `TASKS` 한 곳에 있다. 여기 없는 라벨의 파일은 이름 오타로 보고 건너뛴다.

### 입력 특징

- 플로우 = 5-튜플(출발지·목적지 IP·포트, 프로토콜)로 묶은 패킷 묶음
- 플로우마다 **앞 30개 패킷**의
  - 크기 (페이로드 바이트)
  - 방향 (+1 = 클라이언트→서버, −1 = 서버→클라이언트)
  - 간격 (앞 패킷과의 도착 시간 차)
- CESNET 데이터의 PPI 형식과 같은 모양으로 맞춘다 → 모델 코드 하나로 모든 데이터를 돌린다.
  - 모양 (플로우 수, 3, 30), 행 순서는 간격(ms, 첫 패킷 0) · 방향 · 크기. 30개보다 짧으면 0으로 채운다.
  - TCP는 CESNET처럼 페이로드가 0인 패킷(ACK 등)과 재전송을 빼고, QUIC은 모두 넣는다.
  - 직접 수집 데이터는 443 포트(TLS·QUIC) 플로우만 쓴다.
  - 캡처 전에 열려 중간부터 잡힌 연결은 뺀다 — 앞부분이 잘려 있어서. TCP는 SYN, QUIC은 핸드셰이크(long header) 패킷으로 시작하지 않으면 중간부터 잡힌 것이다. 인스타 화면 분류처럼 앱을 켜 둔 채 캡처를 켤 때 특히 많다.
- **IP·포트·SNI(접속 도메인)는 입력에 넣지 않는다.** 넣으면 모델이 그걸로 답을 외운다.
- 페이로드 바이트는 넣지 않는다 (암호문이라 잡음).

### 모델

1. **1D-CNN** — 기본 모델
2. **작은 Transformer** — 비교 대상
3. 랜덤포레스트 — 기준선 (`--model rf`). 원시 PPI 90개 값을 그대로 넣는다. 딥러닝이 단순한 모델보다 얼마나 나은지 보이려고 결과 표에만 넣고, 프로토타입에는 쓰지 않는다

---

## 3. 데이터

| 데이터 | 역할 | 비고 |
| --- | --- | --- |
| **CESNET-QUIC22** (주) | 큰 규모의 최신 QUIC 데이터로 모델 비교 | 2022년 수집, 앞 30개 패킷의 크기·방향·간격(PPI) 포함. `cesnet-datazoo`로 받는다. 크기 XS·S·M·L 중 **XS로 시작**. pcap 원본 없음 |
| **직접 수집** | 지금 쓰는 실제 앱에서도 되는지 확인, pcap 전처리 모듈 검증, 프로토타입 시연 | PCAPdroid(안드로이드)로 앱별 수집 |
| ISCX VPN-nonVPN 2016 (선택) | 기존 논문과 비교 | 2016년이라 TLS 1.3 이전 트래픽이 대부분. 벅차면 뺀다 — 직접 수집이 pcap 검증 역할을 대신함 |

### CESNET-QUIC22 불러오기

```python
from cesnet_datazoo.datasets import CESNET_QUIC22
from cesnet_datazoo.config import DatasetConfig, AppSelection

dataset = CESNET_QUIC22("D:/datasets/CESNET-QUIC22/", size="XS")
dataset_config = DatasetConfig(
    dataset=dataset,
    apps_selection=AppSelection.ALL_KNOWN,
    train_period_name="W-2022-44",   # 44주차로 학습
    test_period_name="W-2022-45",    # 45주차로 시험
)
dataset.set_dataset_config_and_initialize(dataset_config)
train_df = dataset.get_train_df()
val_df = dataset.get_val_df()
test_df = dataset.get_test_df()
```

학습·시험이 서로 다른 주차라 「날짜 단위로 나누기」가 자동으로 지켜진다.

### 직접 수집 방법 (PCAPdroid)

1. 안드로이드 폰에 PCAPdroid 설치 (루팅 불필요)
2. 설정에서 **대상 앱을 하나만** 선택, TLS 복호화·QUIC 차단은 끈다. 트래픽 덤프는 **PCAP 파일** — UDP 추출기는 패킷이 조용히 빠질 수 있고(앞 30개 패킷이 달라짐), HTTP 서버·TCP 추출기는 캡처할 때마다 PC가 같은 Wi-Fi에서 받고 있어야 한다. 저장 폴더는 `Download/PCAPdroid` (`scripts/pull_pcaps.ps1` 기본값)
   - 라벨마다 네트워크(집 Wi-Fi·LTE 등)를 다르게 쓰지 않는다 — 모델이 앱 대신 네트워크 차이를 배울 수 있다. 같은 날 같은 네트워크에서 라벨 6개를 돌아가며 모은다
3. 위 「분류 대상」 표의 수집 방법대로 5분 사용 → pcap 저장. 파일 하나 = 라벨 하나
4. 라벨 6개, **매일 라벨마다 5분씩 2~3주** (하루 30분). 부담되면 화면 분류 3개는 이틀에 한 번
5. 파일 이름 규칙: `라벨_날짜_순번.pcap` (예: `youtube_1013_1.pcap`, `instagram-reels_1013_1.pcap`)
6. 전처리: pcap → 플로우 → 앞 30개 패킷 (tshark로 CSV 추출 → pandas)
7. 첫날 확인: `python -m src.preprocess`가 라벨별 플로우 수와 「처음 10초」 안에 열린 플로우 수를 보여 준다. 화면 분류 파일에서 처음 몇 초에 플로우가 몰려 있으면 캡처를 켤 때 생긴 재연결일 수 있다 → 학습 때 `--skip-first`로 뺀다
8. 자세히 보기: `notebooks/02_self_collected_explore.ipynb` — `--skip-first` 값, TCP도 쓸지(`--proto`), 라벨별 플로우 수를 여기서 정한다

- 아이폰이면: PC 모바일 핫스팟에 폰을 연결하고 Wireshark로 핫스팟 인터페이스를 캡처, 앱을 하나씩만 켜서 사용
- 유튜브·크롬은 QUIC을 써서 최신 트래픽이 자연스럽게 모인다

---

## 4. 평가 원칙 (가장 많이 틀리는 곳)

- 학습·시험 데이터는 **수집한 날짜(또는 세션) 단위로** 나눈다. 무작위로 섞으면 같은 세션이 양쪽에 들어가 정확도가 부풀려진다.
- 입력에서 **IP·포트·SNI를 뺀다.**
- **정확도 + macro-F1**을 함께 적는다 (앱별 데이터 양이 달라서).
- **85%는 측정 조건을 먼저 정해 둔다** — 어떤 데이터셋, 몇 개 앱, 어떻게 나눴는지. 조건 없이 낸 85%는 의미가 없다.
- 직접 수집 데이터의 앱 분류와 인스타 화면 분류는 따로 학습·평가한다. 계획서의 「응용 분류」는 앱 분류다.
- 결과 표는 데이터셋(CESNET / 직접 수집 앱 분류 / 인스타 화면 분류)별로 CNN vs Transformer.
- 프로토타입 시스템 지표: 초당 처리 플로우 수, 저장량.

---

## 5. 프로토타입

- **Streamlit** — 파이썬 파일 하나로 웹 화면. `streamlit run app.py`
- pcap 업로드 → 플로우별 분류 결과 → 앱별 표·차트
- 왼쪽에서 `runs/`의 학습 결과(모델) 중 하나를 고른다. 플로우는 그 모델을 학습할 때의 조건(`--proto`, `--skip-first`)대로 거른다
- 파일 이름이 `라벨_날짜_순번.pcap`이면 그 라벨을 정답으로 본 정확도도 보여 준다
- 시스템 지표도 함께 보인다: 처리 시간·처리량(초당 플로우 수), 저장량(pcap 대비 특징값 크기 — 플로우당 360바이트)
- 같은 계산을 명령어로: `python -m src.predict <pcap·폴더>` — 여러 파일의 처리량·저장량을 한꺼번에 잰다 (4단계 보고용)
- 시연 입력: 직접 수집한 pcap 중 **학습에 쓰지 않은 날짜분**
- 화면에 시간 쓰지 말고 모델에 집중

---

## 6. 개발 환경

- **VS Code** + 확장 **Python**, **Jupyter** (둘 다 Microsoft)
  - 노트북(.ipynb): 데이터 탐색·그래프
  - 파이썬 파일(.py): 전처리·모델·학습처럼 다시 쓰는 코드
- **Python 3.10 이상** (cesnet-datazoo 요구) — 지금은 3.12로 `.venv` 가상환경을 만들어 쓴다
- **Wireshark** 설치 (tshark 포함 — pcap → CSV)
- GPU 확인: `nvidia-smi`
  - NVIDIA GPU 있음 → 내 PC에서 학습 (PyTorch CUDA 버전 설치). 지금 PC: GTX 1660 Ti 6GB
  - 없음 → CPU로 시작 (모델이 작고 XS면 충분). 느리면 학습만 Google Colab으로

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126   # GPU 있을 때만
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 실행

```powershell
.\.venv\Scripts\python.exe -m pytest                              # 테스트 (가짜 데이터로)
.\scripts\pull_pcaps.ps1                                          # 폰에서 새 pcap 받기 + 수집 현황 (adb 필요)
.\.venv\Scripts\python.exe -m src.preprocess --status             # 수집 현황만 (파일 이름 검사, 앱×날짜 파일 수)
.\.venv\Scripts\python.exe -m src.preprocess                      # 직접 수집 pcap → D:/datasets/processed/self_collected.npz
.\.venv\Scripts\python.exe -m src.train --dataset cesnet --model cnn
.\.venv\Scripts\python.exe -m src.train --dataset self --task app --model transformer --test-dates 1027,1028
.\.venv\Scripts\python.exe -m src.train --dataset self --task screen --model cnn --skip-first 10
.\.venv\Scripts\python.exe -m src.train --dataset self --task app --model cnn --proto quic        # QUIC 플로우만
.\.venv\Scripts\python.exe -m src.train --dataset cesnet --model rf                               # 랜덤포레스트 기준선
.\.venv\Scripts\python.exe -m src.report                                                         # 결과 표 (데이터셋 × 모델, 마크다운)
.\.venv\Scripts\python.exe -m src.predict D:/datasets/self_collected/youtube_1027_1.pcap          # pcap 분류 + 처리량·저장량
.\.venv\Scripts\streamlit.exe run app.py                                                         # 프로토타입
```

학습 결과는 `runs/<시각>_<데이터>_<모델>/`에 남는다 — `metrics.json`(정확도·macro-F1·측정 조건·모델 처리량), `per_class.csv`(앱별 정밀도·재현율·F1), `confusion.csv`·`confusion.png`(혼동 행렬), `model.pt`(가중치).

- 직접 수집 데이터를 학습하려면 날짜가 2일 이상 있어야 한다. 기본값은 마지막 3일을 시험용으로 쓰므로, 처음 며칠은 `--test-dates`로 시험 날짜를 직접 준다.
- `tests/test_tshark.py`는 가짜 pcap을 만들어 진짜 tshark로 전처리 전체를 돌려 본다. tshark가 없는 PC에서는 건너뛴다.

### 폴더 구성

```
cipher_lens/         # 코드 (git 저장소 — C:/Users/jhpark/Documents/GitHub/, OneDrive 밖)
  README.md          # 이 문서
  todo.md            # 할 일
  notebooks/         # 탐색·그래프
    01_cesnet_explore.ipynb          # CESNET 둘러보기
    02_self_collected_explore.ipynb  # 직접 수집 둘러보기 (--skip-first·--proto 정하기)
  src/
    config.py        # 데이터 경로·PPI 상수·과제별 라벨
    preprocess.py    # pcap → 앞 30개 패킷 (간격, 방향, 크기)
    data.py          # CESNET·직접 수집 데이터 불러오기, 날짜 단위 분할
    models.py        # CNN, Transformer
    train.py         # 학습·평가
    predict.py       # 학습한 모델로 pcap 분류, 처리량·저장량
    report.py        # 결과 표, 혼동 행렬 그림
  tests/             # 가짜 데이터로 돌리는 테스트 (test_tshark.py만 tshark 필요)
  scripts/
    pull_pcaps.ps1   # 폰 → PC pcap 받기 (adb)
  app.py             # Streamlit 프로토타입
  runs/              # 학습 결과 (커밋 안 함)
  requirements.txt

D:/datasets/         # 데이터 (저장소 밖, 환경변수 CIPHERLENS_DATA로 바꿀 수 있음)
  CESNET-QUIC22/     # cesnet-datazoo가 받는 곳
  self_collected/    # 직접 수집한 pcap — 바탕 화면의 「pcap」 바로가기가 이 폴더를 연다
  processed/         # 전처리 결과 (npz)
```

- **데이터는 저장소 밖 `D:/datasets/`에 둔다.** 큰 데이터셋과 직접 수집한 pcap이 git 저장소에 섞이지 않게 한다. 바탕 화면은 OneDrive 동기화 폴더라 거기에도 두지 않는다 (클라우드로 올라간다).
- `.gitignore`에 `data/`, `*.pcap`, `*.pcapng`, `*.npz`, `runs/`를 넣어 실수로 커밋하지 않게 한다.

---

## 7. 일정

학기가 12월 중순에 끝난다고 보고 잡았다. 종강일에 맞게 조정.

| 시기 | 단계 |
| --- | --- |
| ~10월 셋째 주 | 1단계: 관련 연구 조사, CESNET 데이터 확보, 직접 수집 시작 |
| 10월 말~11월 초 | 2단계: 전처리·특징 추출 (pcap → 앞 30개 패킷) |
| 11월 | 3단계: CNN·Transformer 학습, 성능 비교 |
| 12월 초 | 4단계: 프로토타입 통합, 검증, 결과 정리 |

---

## 8. 주의사항

- **본인 기기 트래픽만 수집한다.** 공용 와이파이에서 남의 트래픽을 잡으면 통신비밀보호법 위반. PCAPdroid는 그 폰의 트래픽만 본다.
- **pcap 원본은 공개하지 않는다.** 암호화돼 있어도 접속한 서버 주소·도메인 기록이 남는다. 공유가 필요하면 추출한 특징값만.
- **라벨은 완벽하지 않다** — 앱 하나만 골라도 광고·분석 서버 연결이 섞인다. 한계로 적는다.
- **PCAPdroid는 VPN 방식**이라 실제 망 패킷과 크기·타이밍이 조금 다를 수 있다. 한계로 적는다.
- **인스타 화면 라벨은 앱 라벨보다 더 거칠다** — 화면에 들어간 뒤 새로 열린 연결만 남기지만, 앱이 화면과 상관없이 미리 받아 두는 트래픽(다른 화면 콘텐츠 선로딩, 분석 서버)이 섞일 수 있다. 한계로 적는다.

---

## 9. 참고 자료

### 먼저 읽을 논문 3편

1. **DataZoo: Streamlining Traffic Classification Experiments** — CESNET 데이터 쓰는 법 · <https://arxiv.org/abs/2310.19568>
2. **SoK: Decoding the Enigma of Encrypted Network Traffic Classifiers** — 평가에서 흔히 틀리는 부분 · <https://arxiv.org/abs/2503.20093>
3. **Wang et al., 2017, "End-to-end encrypted traffic classification with one-dimensional convolution neural networks"** (IEEE ISI) — 1D-CNN의 출발점

### 더 볼 것 (지름길 학습·평가 함정)

- The Sweet Danger of Sugar: Debunking Representation Learning for Encrypted Traffic Classification · <https://arxiv.org/abs/2507.16438>
- Bias in the Shadows: Explore Shortcuts in Encrypted Network Traffic Classification · <https://arxiv.org/abs/2601.10180>

### 데이터·도구

- cesnet-datazoo · <https://github.com/CESNET/cesnet-datazoo>
- CESNET-QUIC22 (Zenodo) · <https://zenodo.org/record/7409924>
- ISCX VPN-nonVPN 2016 · <https://www.unb.ca/cic/datasets/vpn.html>
- PCAPdroid · <https://github.com/emanuele-f/PCAPdroid>
- Streamlit · <https://streamlit.io>
