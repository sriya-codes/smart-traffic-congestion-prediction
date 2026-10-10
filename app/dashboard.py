import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.traffic_model import generate_demo_data, create_features
st.set_page_config(page_title="Traffic Congestion Predictor", layout="wide")
st.title("🚦 Smart Traffic Congestion Prediction")
st.caption("Machine learning demo using synthetic traffic-speed data.")
data = generate_demo_data()
st.warning("Demo only: these are synthetic observations, not live traffic data.")
latest_speed = float(data["speed_kmh"].iloc[-1])
features = create_features(data)
predicted_speed = latest_speed
if len(features) > 0:
    st.info("Baseline estimate shown below uses the latest observed speed.")
col1, col2 = st.columns(2)
with col1:
    st.metric("Latest observed speed", f"{latest_speed:.1f} km/h")
with col2:
    st.metric("Baseline next-speed estimate", f"{predicted_speed:.1f} km/h")
st.subheader("Recent traffic speeds")
st.line_chart(data.set_index("timestamp")["speed_kmh"])
st.subheader("Interpretation")
if predicted_speed < 20:
    st.error("Possible heavy congestion")
elif predicted_speed < 40:
    st.warning("Possible moderate congestion")
else:
    st.success("Relatively free-flowing traffic")
st.caption("This dashboard is illustrative and does not provide real-time road conditions.")
