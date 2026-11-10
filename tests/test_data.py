import numpy as np
import pytest

from src.config import DIR, IPT, SIZE
from src.data import hide_handshake, load_self_collected, pick_val_sessions, skip_packets, split_by_date


def test_split_by_date_keeps_dates_apart():
    dates = np.array(["1013", "1013", "1014", "1015", "1015"])
    train, test = split_by_date(dates, ["1015"])
    assert set(dates[train]) == {"1013", "1014"}
    assert set(dates[test]) == {"1015"}


def test_split_by_date_rejects_unknown_date():
    with pytest.raises(ValueError):
        split_by_date(np.array(["1013"]), ["1099"])


def test_app_task(npz_path):
    data = load_self_collected(npz_path, task="app", n_test_dates=2, val_frac=0.2)
    assert data.classes == ["instagram", "youtube"]  # 화면 라벨은 섞이지 않는다
    assert data.condition["test_dates"] == ["1016", "1017"]
    assert len(data.test.y) == 2 * 2 * 2 * 4
    assert len(data.train.y) + len(data.val.y) + len(data.test.y) == 80
    assert len(data.val.y) % 4 == 0 and len(data.val.y) > 0  # 세션 단위로 통째로 빠진다
    assert data.train.X.shape[1:] == (3, 30)


def test_screen_task_skips_first_seconds(npz_path):
    data = load_self_collected(npz_path, task="screen", skip_first_s=10)
    assert data.classes == ["instagram-feed", "instagram-reels"]
    assert data.condition["skip_first_s"] == 10
    assert len(data.train.y) + len(data.val.y) + len(data.test.y) == 2 * 5 * 2 * 2  # offset 20·60초만 남는다


def test_explicit_test_dates(npz_path):
    data = load_self_collected(npz_path, test_dates=["1013"])
    assert data.condition["test_dates"] == ["1013"]
    assert "1013" not in data.condition["train_dates"]


def test_proto_filter(npz_path):
    for proto in ["quic", "tcp"]:
        data = load_self_collected(npz_path, proto=proto)
        assert len(data.train.y) + len(data.val.y) + len(data.test.y) == 40  # 앱 라벨 80개 플로우의 절반씩
        assert data.condition["proto"] == proto


def test_labels_and_dates_are_plain_str(npz_path):
    """numpy 문자열이 섞이면 model.pt를 torch.load 기본값으로 다시 불러올 수 없다."""
    data = load_self_collected(npz_path)
    for value in [*data.classes, *data.condition["train_dates"], *data.condition["test_dates"]]:
        assert type(value) is str


def test_val_has_every_label(npz_path):
    """검증 세트에 모든 앱이 들어간다 — 무작위로 세션을 고르면 파일이 적을 때 앱 하나가 통째로 빠질 수 있다."""
    for seed in range(5):
        data = load_self_collected(npz_path, task="app", seed=seed)
        assert set(data.val.y.tolist()) == {0, 1}


def test_pick_val_sessions_keeps_single_session_label_in_train():
    sessions = np.array(["a_1", "a_1", "a_2", "b_1"])
    labels = np.array([0, 0, 0, 1])
    val = pick_val_sessions(sessions, labels, frac=0.1, rng=np.random.default_rng(0))
    assert val.sum() in (1, 2) and not val[3]  # a는 세션 하나를 통째로, b는 세션이 하나뿐이라 학습에 남김
    assert len(set(sessions[val])) == 1


def test_skip_packets_shifts_left_and_pads():
    X = np.zeros((2, 3, 30), dtype=np.float32)
    X[:, IPT] = np.arange(30)  # 간격 0, 1, 2, ...
    X[:, DIR] = 1
    X[:, SIZE] = np.arange(100, 130)
    X[1, :, 4:] = 0  # 두 번째 플로우는 패킷 4개뿐
    out = skip_packets(X, 2)
    np.testing.assert_array_equal(out[0, SIZE, :28], np.arange(102, 130))
    assert out[0, IPT, 0] == 0 and out[0, IPT, 1] == 3  # 새 첫 패킷 간격은 0, 그 뒤는 그대로
    assert not out[:, :, 28:].any()  # 뒤는 0으로 채움
    assert (out[1, DIR] != 0).sum() == 2
    assert not skip_packets(X, 4)[1].any()  # 패킷이 n개 이하면 빈 입력
    np.testing.assert_array_equal(skip_packets(X, 0), X)


def test_hide_handshake_keeps_every_flow(npz_path):
    data = load_self_collected(npz_path, task="app", n_test_dates=2)
    hidden = hide_handshake(data, 5)
    for before, after in ((data.train, hidden.train), (data.val, hidden.val), (data.test, hidden.test)):
        assert len(after.y) == len(before.y)  # 시험 세트가 가리지 않은 결과와 같다
        np.testing.assert_array_equal(after.X, skip_packets(before.X, 5))
    assert hidden.condition["skip_packets"] == 5
    assert hidden.condition["empty_after_skip"] == {"train": 0, "val": 0, "test": 0}
    assert "skip_packets" not in data.condition  # 원래 Data는 그대로


def test_too_few_dates_for_training(npz_path):
    with pytest.raises(ValueError, match="--test-dates"):
        load_self_collected(npz_path, n_test_dates=5)  # 수집 날짜 5일이 모두 시험으로 가면 학습할 날이 없다
