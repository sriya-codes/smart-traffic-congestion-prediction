"""Traffic-speed forecasting baseline. Demo observations are synthetic."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURE_COLUMNS = ["lag_1", "lag_2", "lag_3", "rolling_mean_3", "hour", "day_of_week"]


def generate_demo_data(periods=24 * 14, seed=42):
    """Generate reproducible synthetic five-minute speed observations."""
    if periods < 20:
        raise ValueError("periods must be at least 20")
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-06", periods=periods, freq="5min")
    minute = timestamps.hour * 60 + timestamps.minute
    morning = np.exp(-((minute - 480) / 95) ** 2)
    evening = np.exp(-((minute - 1050) / 115) ** 2)
    daily = 4 * np.sin(2 * np.pi * minute / 1440)
    noise = rng.normal(0, 3, periods)
    speed = 62 - 24 * (morning + evening) + daily + noise
    return pd.DataFrame({"timestamp": timestamps, "speed_kmh": np.clip(speed, 8, 80).round(2)})


def create_features(data):
    """Create lag and calendar features after sorting observations chronologically."""
    required = {"timestamp", "speed_kmh"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = data[["timestamp", "speed_kmh"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame["speed_kmh"] = pd.to_numeric(frame["speed_kmh"], errors="raise")
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp")
    if frame["speed_kmh"].isna().any() or not np.isfinite(frame["speed_kmh"]).all():
        raise ValueError("speed_kmh must contain finite, non-missing values")
    if (frame["speed_kmh"] < 0).any():
        raise ValueError("speed_kmh cannot be negative")
    for lag in (1, 2, 3):
        frame[f"lag_{lag}"] = frame["speed_kmh"].shift(lag)
    frame["rolling_mean_3"] = frame["speed_kmh"].shift(1).rolling(3).mean()
    frame["hour"] = frame["timestamp"].dt.hour
    frame["day_of_week"] = frame["timestamp"].dt.dayofweek
    return frame.dropna().reset_index(drop=True)


def train_baseline(data, test_fraction=0.2):
    """Train Random Forest with a chronological holdout; never shuffle time."""
    if not 0.05 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.05 and 0.5")
    frame = create_features(data)
    if len(frame) < 30:
        raise ValueError("Not enough observations after feature creation")
    split = int(len(frame) * (1 - test_fraction))
    train, test = frame.iloc[:split], frame.iloc[split:]
    model = RandomForestRegressor(
        n_estimators=150, max_depth=12, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    model.fit(train[FEATURE_COLUMNS], train["speed_kmh"])
    predictions = np.clip(model.predict(test[FEATURE_COLUMNS]), 0, None)
    mae = float(mean_absolute_error(test["speed_kmh"], predictions))
    rmse = float(np.sqrt(mean_squared_error(test["speed_kmh"], predictions)))
    return model, mae, rmse, len(test)


def next_step_features(data):
    """Build features for a five-minute-ahead estimate from latest observations."""
    frame = data[["timestamp", "speed_kmh"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = frame.sort_values("timestamp")
    if len(frame) < 3:
        raise ValueError("At least three observations are required")
    speeds = frame["speed_kmh"].astype(float).to_numpy()
    if not np.isfinite(speeds[-3:]).all() or (speeds[-3:] < 0).any():
        raise ValueError("Latest three speeds must be finite and non-negative")
    next_time = frame["timestamp"].iloc[-1] + pd.Timedelta(minutes=5)
    return pd.DataFrame([{
        "lag_1": speeds[-1], "lag_2": speeds[-2], "lag_3": speeds[-3],
        "rolling_mean_3": float(np.mean(speeds[-3:])),
        "hour": next_time.hour, "day_of_week": next_time.dayofweek,
    }], columns=FEATURE_COLUMNS)
