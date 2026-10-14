import numpy as np
import pytest

from src.data import load_self_collected, split_by_date


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


def test_too_few_dates_for_training(npz_path):
    with pytest.raises(ValueError, match="--test-dates"):
        load_self_collected(npz_path, n_test_dates=5)  # 수집 날짜 5일이 모두 시험으로 가면 학습할 날이 없다
