# TODO

자세한 내용은 [README.md](./README.md). 일정은 학기가 12월 중순에 끝난다고 보고 잡았다.

## 확인할 것

- [ ] 지도교수님께 데이터 구성 확인 메일 — 「공개 데이터(CESNET-QUIC22) + 직접 수집, 회사 데이터는 쓰지 않음」
- [ ] 하잎아이피에 회사 데이터를 쓰지 않는다고 알리기 + 회사 처리·저장 환경을 쓸지 정하기
- [ ] 종강일 확인 → 아래 일정 조정
- [ ] 85% 측정 조건 정하기 (데이터셋 · 앱 수 · 나누는 방법)
- [ ] 파일당 플로우 수 확인 — 인스타는 QUIC 연결 몇 개에 트래픽을 몰아서, 첫 파일(약 3분)에서 남은 플로우가 8개뿐이었다. 강제 종료 후 다시 재 보고, 적으면 「5분 1번」 대신 「짧게 여러 번(매번 강제 종료 후)」으로 바꾼다 (플로우 수는 사용 시간보다 앱을 새로 켠 횟수에 비례)
- [ ] 화면 분류(`screen`)가 될지 확인 — 앱을 켠 채 캡처하므로 기존 연결은 모두 빠지고, 화면에 들어간 뒤 새로 열린 연결만 남는다. 첫 화면 분류 파일에서 남는 플로우가 거의 없으면 방법을 다시 정한다
- [ ] ISCX VPN-nonVPN 2016 쓸지 정하기 (벅차면 뺀다)

## 이번 주 (10/13 ~ 10/18)

- [x] 분류 대상 정하기 — 앱 분류: 유튜브·네이버웹툰·인스타 / 인스타 화면 분류: 릴스·피드·탐색 탭 (README 2절)
- [ ] PCAPdroid 설치
- [ ] 직접 수집 시작 — 매일 라벨 6개 × 5분 (하루 30분), 파일명 `라벨_날짜_순번.pcap` (2~3주 계속)
- [ ] 첫날 수집 후 `python -m src.preprocess` 실행 → 라벨별 플로우 수, 화면 분류 파일의 「처음 10초」 플로우 확인 → `--skip-first` 값 정하기 — `notebooks/02_self_collected_explore.ipynb` ①
- [ ] VS Code 확장 설치: Python, Jupyter
- [x] Python 3.10 이상 확인, 가상환경 만들기 — Python 3.12, `.venv`
- [x] `pip install cesnet-datazoo torch scikit-learn pandas matplotlib streamlit` — `requirements.txt`
- [x] Wireshark 설치 (tshark 포함) — 4.6.9. `tests/test_tshark.py`(가짜 pcap → 진짜 tshark → 전처리) 통과
- [ ] (선택) adb 설치 (Android SDK Platform-Tools) + 폰 USB 디버깅 — `scripts/pull_pcaps.ps1`로 폰 → PC
- [x] `nvidia-smi`로 GPU 확인 → GPU 있으면 PyTorch CUDA 버전으로 재설치 — GTX 1660 Ti, torch cu126
- [x] 폴더 구성 만들기 — 코드는 `cipher_lens/`(`notebooks/` `src/`), 데이터는 **OneDrive 밖** `D:/datasets/`(`CESNET-QUIC22/` `self_collected/`)
- [x] `.gitignore`에 `data/` `*.pcap` `*.pcapng`
- [ ] CESNET-QUIC22 XS 받아서 열어 보기 — `notebooks/01_cesnet_explore.ipynb` 실행 (처음 실행 때 내려받음). 10/14 배포 서버(liberouter.org)가 503·무응답이라 못 받음 — 다시 시도. 오래 막히면 Zenodo 원본(21GB zip, 형식이 달라 로더를 새로 짜야 함)
- [ ] **이번 주 목표**: 앱 3~4개의 앞 30개 패킷 크기 시퀀스를 그래프로 그려 보기 — 같은 노트북에 그래프 셀 있음

## 1단계 — 관련 연구 조사 · 데이터 확보 (~10월 셋째 주)

- [ ] 논문: DataZoo
- [ ] 논문: SoK: Decoding the Enigma of Encrypted Network Traffic Classifiers
- [ ] 논문: Wang et al. 2017 (1D-CNN)
- [ ] CESNET-QUIC22 데이터 구조 파악 (PPI 열, 앱 라벨, 주차별 분할) — PPI 행 순서·라벨 인코딩은 라이브러리 소스로 확인함 (CLAUDE.md)
- [ ] 직접 수집 진행 상황 점검 (라벨별 파일 수, 날짜 분포) — `python -m src.preprocess --status`

## 2단계 — 전처리 · 특징 추출 (10월 말 ~ 11월 초)

- [ ] `src/preprocess.py`: pcap → tshark CSV → 플로우(5-튜플)로 묶기 — 코드·가짜 데이터 테스트 있음, 실제 pcap으로 확인 필요
- [ ] 플로우마다 앞 30개 패킷의 (크기, 방향 ±1, 간격) 행렬 만들기 — CESNET PPI와 같은 모양 (위와 같음)
- [x] IP·포트·SNI가 입력에 들어가지 않는지 확인 — `test_ppi_does_not_depend_on_ip_or_port` (IP·포트를 바꿔도 입력이 같음). 전처리를 바꿀 때마다 `pytest`
- [ ] 직접 수집 데이터를 날짜 단위로 학습/시험 분할 — `src/data.py` 작성됨
- [ ] 직접 수집에서 TCP(TLS) 플로우도 쓸지, QUIC만 쓸지 정하기 (CESNET은 QUIC만) — 노트북 02 ②로 라벨별 비율을 보고, `--proto all`과 `--proto quic` 결과 비교
- [x] 실제 pcap에서 QUIC 중간 연결 제외가 맞게 되는지 확인 (`quic.header_form`이 tshark에서 제대로 나오는지) — `instagram_1014_1.pcap`(PCAPdroid, Raw IP): long header 14개를 모두 인식(QUIC v1), 첫 패킷이 short header인 연결 6개는 캡처 전에 열린 것이라 빠짐. 가짜 pcap 테스트는 `tests/test_tshark.py`
- [ ] 직접 수집 데이터 통계 정리 (라벨별 플로우 수) — 노트북 02

## 3단계 — 모델 학습 · 성능 비교 (11월)

- [ ] `src/models.py`: 1D-CNN — 뼈대 작성됨, 실제 데이터로 학습 전
- [ ] `src/models.py`: 작은 Transformer — 뼈대 작성됨, 실제 데이터로 학습 전
- [ ] `src/train.py`: 학습·평가 (정확도 + macro-F1, 혼동 행렬) — 뼈대 작성됨(앱별 지표 `per_class.csv`, 혼동 행렬 그림 `confusion.png` 포함), 실제 데이터로 학습 전
- [ ] CESNET-QUIC22로 CNN vs Transformer
- [ ] 직접 수집 앱 분류(`--task app`)로 CNN vs Transformer
- [ ] 직접 수집 인스타 화면 분류(`--task screen`)로 CNN vs Transformer
- [ ] 결과 표 정리 (CESNET / 앱 분류 / 인스타 화면 분류 × 모델별) — `python -m src.report`가 runs/에서 마크다운 표를 만든다
- [ ] (선택) 랜덤포레스트 기준선 — 코드 있음 (`--model rf`), 실제 데이터로 돌리기 전
- [ ] (선택) ISCX VPN-nonVPN 2016 비교

## 4단계 — 프로토타입 통합 · 검증 · 정리 (12월 초)

- [ ] `app.py` (Streamlit): pcap 업로드 → 플로우별 분류 → 앱별 표·차트 — 작성됨(가짜 데이터 테스트), 실제 모델·pcap으로 확인 필요
- [ ] 시연용 pcap 준비 — 직접 수집 중 학습에 쓰지 않은 날짜분
- [ ] 처리량 측정 (초당 처리 플로우 수), 저장량 기록 — `python -m src.predict <시연 pcap 폴더>`(pcap 전체), `metrics.json`의 `infer_flows_per_s`(모델만)
- [ ] 한계 정리 (라벨 노이즈, 인스타 화면 라벨의 선로딩 트래픽, PCAPdroid VPN 방식, 데이터 규모)
- [ ] 최종 보고서
- [ ] 발표 자료 · 포스터 (정식 과제명 + 부제)
