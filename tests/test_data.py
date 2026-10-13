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


@pytest.fixture
def npz_path(tmp_path):
    """라벨 4개(앱 2 + 화면 2) × 5일 × 하루 2세션 × 세션당 플로우 4개 (offset 0, 5, 20, 60초)."""
    labels, dates, sessions, offsets = [], [], [], []
    for label in ["youtube", "instagram", "instagram-reels", "instagram-feed"]:
        for day in ["1013", "1014", "1015", "1016", "1017"]:
            for seq in [1, 2]:
                for offset in [0.0, 5.0, 20.0, 60.0]:
                    labels.append(label)
                    dates.append(day)
                    sessions.append(f"{label}_{day}_{seq}")
                    offsets.append(offset)
    n = len(labels)
    path = tmp_path / "self_collected.npz"
    np.savez(path, ppi=np.random.rand(n, 3, 30).astype(np.float32), label=np.array(labels), date=np.array(dates),
             session=np.array(sessions), proto=np.array(["udp"] * n), offset=np.array(offsets))
    return path


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
