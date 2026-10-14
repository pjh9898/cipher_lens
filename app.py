"""CipherLens 프로토타입 — pcap을 올리면 플로우마다 어떤 앱인지 분류해 보여 준다.

    streamlit run app.py

tshark가 필요하고, runs/에 python -m src.train으로 학습한 모델이 있어야 한다.
시연 입력은 학습에 쓰지 않은 날짜의 직접 수집 pcap.
"""
import tempfile
from pathlib import Path

import streamlit as st

from src.predict import FEATURE_BYTES_PER_FLOW, classify_pcap, list_runs, load_classifier, summarize

st.set_page_config(page_title="CipherLens", layout="wide")
st.title("CipherLens")
st.caption("응용레벨 암호화 트래픽 AI기반 분석 기술 개발 — 복호화 없이 패킷 패턴으로 TLS 1.3·QUIC 응용을 분류하는 딥러닝 모델")


@st.cache_resource
def classifier(run: str):
    return load_classifier(Path(run))


@st.cache_data(max_entries=5)
def classify(run: str, name: str, data: bytes):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name  # 파일 이름(라벨_날짜_순번)으로 정답을 알 수 있게 이름을 그대로 둔다
        path.write_bytes(data)
        return classify_pcap(classifier(run), path)


runs = list_runs()
if not runs:
    st.warning("runs/에 학습한 모델이 없다. 먼저 `python -m src.train`으로 학습한다.")
    st.stop()

run = st.sidebar.selectbox("모델", runs, format_func=lambda p: p.name)
clf = classifier(str(run))
if clf.metrics:
    st.sidebar.write(f"시험 정확도 **{clf.metrics['accuracy']:.1%}** · macro-F1 **{clf.metrics['macro_f1']:.1%}**")
st.sidebar.write(f"분류하는 앱: {', '.join(clf.classes) if len(clf.classes) <= 10 else f'{len(clf.classes)}개'}")
with st.sidebar.expander("학습 조건"):
    st.json(clf.condition)

uploaded = st.file_uploader("pcap 파일", type=["pcap", "pcapng"])
st.caption("플로우마다 앞 30개 패킷의 크기·방향·간격만 본다. IP·포트·SNI·페이로드는 쓰지 않는다.")
if uploaded is None:
    st.stop()

try:
    result = classify(str(run), uploaded.name, uploaded.getvalue())
except FileNotFoundError as e:  # tshark 없음
    st.error(str(e))
    st.stop()
if result.flows.empty:
    st.warning("분류할 플로우가 없다 (443 포트, 핸드셰이크부터 잡힌 연결만 쓴다).")
    st.stop()

cols = st.columns(4)
cols[0].metric("플로우", f"{len(result.flows):,}")
cols[1].metric("처리 시간", f"{result.total_seconds:.2f}초")
cols[2].metric("처리량", f"{result.flows_per_s:,.0f} 플로우/초")
cols[3].metric("저장량 (pcap 대비)", f"{result.feature_bytes / result.pcap_bytes:.2%}",
               help=f"pcap {result.pcap_bytes / 1e6:,.2f}MB → 특징값 {result.feature_bytes / 1e3:,.1f}KB "
                    f"(플로우당 {FEATURE_BYTES_PER_FLOW}바이트, float32 3×30)")
if result.accuracy is not None:
    st.info(f"파일 이름의 라벨 **{result.true_label}** 기준 정확도 **{result.accuracy:.1%}** — 광고·분석 서버 연결도 섞여 있어 100%가 되기 어렵다.")

summary = summarize(result.flows)
left, right = st.columns(2)
left.subheader("앱별 플로우")
left.dataframe(summary.style.format({"비율": "{:.1%}", "평균 확신도": "{:.2f}"}))
right.bar_chart(summary["플로우"], horizontal=True)

with st.expander("플로우별 결과"):
    flows = result.flows.assign(proto=result.flows["proto"].map({"udp": "QUIC", "tcp": "TLS"}))
    st.dataframe(flows.rename(columns={"offset": "시작(초)", "proto": "프로토콜", "ppi_len": "패킷 수", "app": "예측",
                                       "confidence": "확신도"}).style.format({"시작(초)": "{:.1f}", "확신도": "{:.2f}"}))
st.caption(f"단계별 시간: tshark {result.seconds['tshark']:.2f}초 · 플로우 묶기 {result.seconds['flows']:.2f}초 · "
           f"모델 {result.seconds['model']:.2f}초 ({clf.device})")
