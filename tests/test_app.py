"""프로토타입 화면이 오류 없이 그려지는지 본다. 분류 자체는 test_predict에서."""
from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

import src.predict as predict

APP = str(Path(__file__).resolve().parents[1] / "app.py")


class FakeUpload:
    """AppTest는 파일 업로드를 흉내 내지 못해서 st.file_uploader가 이걸 돌려주게 한다."""
    name = "youtube_1027_1.pcap"

    def getvalue(self) -> bytes:
        return bytes(1000)


def test_app_without_model(monkeypatch):
    monkeypatch.setattr(predict, "list_runs", lambda: [])
    at = AppTest.from_file(APP).run(timeout=60)
    assert not at.exception
    assert "학습한 모델이 없다" in at.warning[0].value


def test_app_with_model(monkeypatch, make_run):
    run = make_run(condition={"dataset": "self_collected", "task": "app"})
    monkeypatch.setattr(predict, "list_runs", lambda: [run])
    at = AppTest.from_file(APP).run(timeout=60)
    assert not at.exception
    assert at.sidebar.selectbox[0].value == run
    assert "instagram, webtoon, youtube" in at.sidebar.markdown[1].value


def test_app_shows_results(monkeypatch, make_run, packets):
    run = make_run()
    monkeypatch.setattr(predict, "list_runs", lambda: [run])
    monkeypatch.setattr(predict, "run_tshark", lambda pcap: "")
    monkeypatch.setattr(predict, "packets_from_tshark", lambda csv_text: packets.copy())
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: FakeUpload())
    at = AppTest.from_file(APP).run(timeout=60)
    assert not at.exception
    assert [m.label for m in at.metric] == ["플로우", "처리 시간", "처리량", "저장량 (pcap 대비)"]
    assert at.metric[0].value == "2"
    assert "youtube" in at.info[0].value  # 파일 이름의 라벨 기준 정확도
