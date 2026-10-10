import pandas as pd
import pytest
from src.traffic_model import create_features, generate_demo_data, next_step_features


def test_demo_data_is_reproducible():
    pd.testing.assert_frame_equal(generate_demo_data(50), generate_demo_data(50))


def test_feature_creation_drops_three_initial_rows():
    assert len(create_features(generate_demo_data(50))) == 47


def test_missing_required_column_raises():
    with pytest.raises(ValueError, match="Missing required columns"):
        create_features(pd.DataFrame({"speed_kmh": [20, 21, 22]}))


def test_next_step_uses_latest_observations():
    data = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=3, freq="5min"),
        "speed_kmh": [30, 20, 10],
    })
    row = next_step_features(data).iloc[0]
    assert row["lag_1"] == 10
    assert row["lag_2"] == 20
    assert row["lag_3"] == 30
    assert row["rolling_mean_3"] == 20


def test_negative_speed_is_rejected():
    data = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=35, freq="5min"),
        "speed_kmh": [20] * 34 + [-1],
    })
    with pytest.raises(ValueError, match="cannot be negative"):
        create_features(data)
