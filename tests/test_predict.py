import numpy as np
import pandas as pd
import pytest

import src.predict as predict
from src.config import N_PACKETS
from src.models import MODELS, build_model
from src.predict import classify_pcap, expand_paths, list_runs, load_classifier, predict_proba, summarize


@pytest.mark.parametrize("name", MODELS)
def test_predict_proba(name):
    model = build_model(name, 4)
    X = np.random.default_rng(0).random((10, 3, N_PACKETS), dtype=np.float32)
    proba = predict_proba(model, X, "cpu", batch_size=3)
    assert proba.shape == (10, 4)
    np.testing.assert_allclose(proba.sum(1), 1, rtol=1e-5)
    assert predict_proba(model, X[:0], "cpu").shape == (0, 4)  # 플로우가 하나도 없는 pcap


def test_list_runs_newest_first(make_run):
    older = make_run("20261101-120000_self-app_cnn")
    newer = make_run("20261102-090000_cesnet_cnn")
    (older.parent / "20261103-000000_broken").mkdir()  # model.pt 없음 → 빠진다
    assert list_runs(older.parent) == [newer, older]


@pytest.fixture
def fake_tshark(monkeypatch, packets):
    """tshark 대신 conftest의 가짜 패킷. 남는 플로우는 TCP(offset 0초) · QUIC(offset 0.3초) 2개."""
    monkeypatch.setattr(predict, "run_tshark", lambda pcap: "")
    monkeypatch.setattr(predict, "packets_from_tshark", lambda csv_text: packets.copy())


def test_classify_pcap(tmp_path, make_run, fake_tshark):
    clf = load_classifier(make_run(), device="cpu")  # torch.load 기본값(weights_only)으로 불러와진다
    pcap = tmp_path / "youtube_1027_1.pcap"
    pcap.write_bytes(bytes(1000))
    result = classify_pcap(clf, pcap)
    assert list(result.flows.columns) == ["offset", "proto", "ppi_len", "app", "confidence"]
    assert len(result.flows) == 2
    assert set(result.flows["app"]) <= set(clf.classes)
    assert result.flows["confidence"].between(0, 1).all()
    assert result.true_label == "youtube"
    assert result.accuracy == (result.flows["app"] == "youtube").mean()
    assert result.feature_bytes == 2 * 3 * N_PACKETS * 4
    assert result.pcap_bytes == 1000
    assert set(result.seconds) == {"tshark", "flows", "model"}


def test_classify_pcap_filters_like_training(tmp_path, make_run, fake_tshark):
    pcap = tmp_path / "capture.pcap"
    pcap.touch()
    quic_only = load_classifier(make_run("a", {"proto": "quic"}), device="cpu")
    assert list(classify_pcap(quic_only, pcap).flows["proto"]) == ["udp"]
    skip_first = load_classifier(make_run("b", {"skip_first_s": 0.1}), device="cpu")
    result = classify_pcap(skip_first, pcap)
    assert list(result.flows["offset"]) == pytest.approx([0.3])
    assert result.true_label is None and result.accuracy is None  # 이름 규칙에 맞지 않는 파일


def test_no_accuracy_when_label_is_not_a_model_class(tmp_path, make_run, fake_tshark):
    clf = load_classifier(make_run(classes=("google-www", "youtube-cesnet")), device="cpu")
    pcap = tmp_path / "webtoon_1027_1.pcap"
    pcap.touch()
    assert classify_pcap(clf, pcap).accuracy is None


def test_summarize():
    flows = pd.DataFrame({"app": ["a", "b", "b", "b"], "confidence": [0.5, 0.9, 0.7, 0.8]})
    summary = summarize(flows)
    assert list(summary.index) == ["b", "a"]
    assert list(summary["플로우"]) == [3, 1]
    np.testing.assert_allclose(summary["비율"], [0.75, 0.25])
    np.testing.assert_allclose(summary["평균 확신도"], [0.8, 0.5])


def test_expand_paths(tmp_path):
    for name in ["youtube_1027_1.pcap", "webtoon_1027_1.pcap", "youtube_1028_1.pcapng", "notes.txt"]:
        (tmp_path / name).touch()
    folder = expand_paths([str(tmp_path)])
    assert [p.name for p in folder] == ["webtoon_1027_1.pcap", "youtube_1027_1.pcap", "youtube_1028_1.pcapng"]
    pattern = expand_paths([str(tmp_path / "*_1027_*.pcap")])
    assert [p.name for p in pattern] == ["webtoon_1027_1.pcap", "youtube_1027_1.pcap"]
    assert expand_paths([str(tmp_path / "youtube_1027_1.pcap")]) == [tmp_path / "youtube_1027_1.pcap"]
