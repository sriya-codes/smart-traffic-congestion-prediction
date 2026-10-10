Smart Traffic Congestion Prediction

A machine learning project that analyzes traffic-speed data to estimate road speeds and understand congestion patterns. It includes a model-training pipeline and an interactive dashboard.

Project Overview

Traffic congestion affects travel time, road efficiency, and daily commuting. This project demonstrates how machine learning can be used to analyze historical traffic observations and estimate future road speeds.

Features

* Traffic data preprocessing and feature engineering
* Time-based features, lag features, and rolling statistics
* Random Forest regression model
* Chronological training and evaluation
* MAE and RMSE model evaluation metrics
* Interactive Streamlit dashboard

Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib

Project Structure

smart-traffic-congestion-prediction/
├── app/
│   └── dashboard.py
├── data/
│   └── README.md
├── reports/
│   └── model_evaluation.md
├── src/
│   └── traffic_model.py
├── tests/
│   └── test_traffic_model.py
├── requirements.txt
├── train.py
└── README.md

Getting Started

1. Install dependencies

pip install -r requirements.txt

2. Train the model

python train.py

3. Launch the dashboard

streamlit run app/dashboard.py

Evaluation

Model performance is assessed using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). See reports/model_evaluation.md for evaluation details.

Data Disclaimer

The initial demo uses synthetic traffic-speed observations. These are illustrative values and do not represent measurements from a real city or road network. Real-world performance requires a suitable, documented traffic dataset.

Future Improvements

* Integrate real traffic datasets
* Compare multiple machine learning models
* Add location-based congestion analysis
* Improve dashboard visualizations
