# TODO

자세한 내용은 [README.md](./README.md). 일정은 학기가 12월 중순에 끝난다고 보고 잡았다.

## 확인할 것

- [ ] 지도교수님께 데이터 구성 확인 메일 — 「공개 데이터(CESNET-QUIC22) + 직접 수집, 회사 데이터는 쓰지 않음」
- [ ] 하잎아이피에 회사 데이터를 쓰지 않는다고 알리기 + 회사 처리·저장 환경을 쓸지 정하기
- [ ] 종강일 확인 → 아래 일정 조정
- [ ] 85% 측정 조건 정하기 (데이터셋 · 앱 수 · 나누는 방법)
- [ ] ISCX VPN-nonVPN 2016 쓸지 정하기 (벅차면 뺀다)

## 이번 주 (10/13 ~ 10/18)

- [ ] PCAPdroid 설치, 수집할 앱 5개 정하기
- [ ] 직접 수집 시작 — 매일 앱마다 5분, 파일명 `앱_날짜_순번.pcap` (2~3주 계속)
- [ ] VS Code 확장 설치: Python, Jupyter
- [x] Python 3.10 이상 확인, 가상환경 만들기 — Python 3.12, `.venv`
- [x] `pip install cesnet-datazoo torch scikit-learn pandas matplotlib streamlit` — `requirements.txt`
- [ ] Wireshark 설치 (tshark 포함)
- [ ] (선택) adb 설치 (Android SDK Platform-Tools) + 폰 USB 디버깅 — `scripts/pull_pcaps.ps1`로 폰 → PC
- [x] `nvidia-smi`로 GPU 확인 → GPU 있으면 PyTorch CUDA 버전으로 재설치 — GTX 1660 Ti, torch cu126
- [x] 폴더 구성 만들기 — 코드는 `cipher_lens/`(`notebooks/` `src/`), 데이터는 **OneDrive 밖** `D:/datasets/`(`CESNET-QUIC22/` `self_collected/`)
- [x] `.gitignore`에 `data/` `*.pcap` `*.pcapng`
- [ ] CESNET-QUIC22 XS 받아서 열어 보기 — `notebooks/01_cesnet_explore.ipynb` 실행 (처음 실행 때 내려받음)
- [ ] **이번 주 목표**: 앱 3~4개의 앞 30개 패킷 크기 시퀀스를 그래프로 그려 보기 — 같은 노트북에 그래프 셀 있음

## 1단계 — 관련 연구 조사 · 데이터 확보 (~10월 셋째 주)

- [ ] 논문: DataZoo
- [ ] 논문: SoK: Decoding the Enigma of Encrypted Network Traffic Classifiers
- [ ] 논문: Wang et al. 2017 (1D-CNN)
- [ ] CESNET-QUIC22 데이터 구조 파악 (PPI 열, 앱 라벨, 주차별 분할) — PPI 행 순서·라벨 인코딩은 라이브러리 소스로 확인함 (CLAUDE.md)
- [ ] 직접 수집 진행 상황 점검 (앱별 파일 수, 날짜 분포) — `python -m src.preprocess --status`

## 2단계 — 전처리 · 특징 추출 (10월 말 ~ 11월 초)

- [ ] `src/preprocess.py`: pcap → tshark CSV → 플로우(5-튜플)로 묶기 — 코드·가짜 데이터 테스트 있음, 실제 pcap으로 확인 필요
- [ ] 플로우마다 앞 30개 패킷의 (크기, 방향 ±1, 간격) 행렬 만들기 — CESNET PPI와 같은 모양 (위와 같음)
- [ ] IP·포트·SNI가 입력에 들어가지 않는지 확인
- [ ] 직접 수집 데이터를 날짜 단위로 학습/시험 분할 — `src/data.py` 작성됨
- [ ] 직접 수집에서 TCP(TLS) 플로우도 쓸지, QUIC만 쓸지 정하기 (CESNET은 QUIC만)
- [ ] 직접 수집 데이터 통계 정리 (앱별 플로우 수)

## 3단계 — 모델 학습 · 성능 비교 (11월)

- [ ] `src/models.py`: 1D-CNN — 뼈대 작성됨, 실제 데이터로 학습 전
- [ ] `src/models.py`: 작은 Transformer — 뼈대 작성됨, 실제 데이터로 학습 전
- [ ] `src/train.py`: 학습·평가 (정확도 + macro-F1, 혼동 행렬) — 뼈대 작성됨, 실제 데이터로 학습 전
- [ ] CESNET-QUIC22로 CNN vs Transformer
- [ ] 직접 수집 데이터로 CNN vs Transformer
- [ ] 결과 표 정리 (데이터셋별 × 모델별)
- [ ] (선택) 랜덤포레스트 기준선
- [ ] (선택) ISCX VPN-nonVPN 2016 비교

## 4단계 — 프로토타입 통합 · 검증 · 정리 (12월 초)

- [ ] `app.py` (Streamlit): pcap 업로드 → 플로우별 분류 → 앱별 표·차트
- [ ] 시연용 pcap 준비 — 직접 수집 중 학습에 쓰지 않은 날짜분
- [ ] 처리량 측정 (초당 처리 플로우 수), 저장량 기록
- [ ] 한계 정리 (라벨 노이즈, PCAPdroid VPN 방식, 데이터 규모)
- [ ] 최종 보고서
- [ ] 발표 자료 · 포스터 (정식 과제명 + 부제)
