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
    """앱 2개 × 5일 × 하루 2세션 × 세션당 플로우 4개."""
    apps, dates, sessions = [], [], []
    for app in ["youtube", "instagram"]:
        for day in ["1013", "1014", "1015", "1016", "1017"]:
            for seq in [1, 2]:
                for _ in range(4):
                    apps.append(app)
                    dates.append(day)
                    sessions.append(f"{app}_{day}_{seq}")
    n = len(apps)
    path = tmp_path / "self_collected.npz"
    np.savez(path, ppi=np.random.rand(n, 3, 30).astype(np.float32), app=np.array(apps), date=np.array(dates),
             session=np.array(sessions), proto=np.array(["udp"] * n))
    return path


def test_load_self_collected(npz_path):
    data = load_self_collected(npz_path, n_test_dates=2, val_frac=0.2)
    assert data.classes == ["instagram", "youtube"]
    assert data.condition["test_dates"] == ["1016", "1017"]
    assert len(data.test.y) == 2 * 2 * 2 * 4
    assert len(data.train.y) + len(data.val.y) + len(data.test.y) == 80
    assert len(data.val.y) % 4 == 0 and len(data.val.y) > 0  # 세션 단위로 통째로 빠진다
    assert data.train.X.shape[1:] == (3, 30)


def test_load_self_collected_explicit_test_dates(npz_path):
    data = load_self_collected(npz_path, test_dates=["1013"])
    assert data.condition["test_dates"] == ["1013"]
    assert "1013" not in data.condition["train_dates"]
