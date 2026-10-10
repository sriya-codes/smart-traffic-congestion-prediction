"""Streamlit dashboard for the traffic-speed forecasting prototype."""
from pathlib import Path
import sys
import joblib
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.traffic_model import generate_demo_data, next_step_features  # noqa: E402

MODEL_PATH = ROOT / "models" / "traffic_model.joblib"
st.set_page_config(page_title="Traffic Forecast", page_icon="🚦", layout="wide")
st.title("🚦 Smart Traffic Congestion Prediction")
st.caption("Portfolio prototype. Synthetic demo data is used unless a real dataset is supplied.")

if not MODEL_PATH.exists():
    st.warning("No trained model found. Run `python train.py` from the project root, then restart the dashboard.")
    st.stop()

model = joblib.load(MODEL_PATH)
data = generate_demo_data()
prediction = max(0.0, float(model.predict(next_step_features(data))[0]))
col1, col2, col3 = st.columns(3)
col1.metric("Latest illustrative speed", f"{data['speed_kmh'].iloc[-1]:.1f} km/h")
col2.metric("Next 5-minute estimate", f"{prediction:.1f} km/h")
category = "High congestion" if prediction < 20 else "Moderate congestion" if prediction < 40 else "Free-flowing (illustrative)"
col3.metric("Illustrative category", category)
recent = data.tail(24 * 12)
fig = px.line(recent, x="timestamp", y="speed_kmh",
              labels={"timestamp": "Time", "speed_kmh": "Speed (km/h)"},
              title="Synthetic speed series — last 24 hours")
st.plotly_chart(fig, use_container_width=True)
st.info("This demo uses generated synthetic observations. It is not connected to live traffic feeds and must not be used for real travel decisions.")
