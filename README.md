# Smart Traffic Congestion Prediction Using ML

A Python machine-learning prototype that estimates near-term traffic speed from recent observations and calendar features. Includes reproducible synthetic demo data, chronological evaluation, a training CLI, automated tests, and a Streamlit dashboard.

> **Data honesty:** The built-in dataset is synthetic and illustrative. This repository is not connected to live traffic feeds. Do not interpret demo results as real-world traffic performance.

## Features
- Synthetic demo data generator for a first run without downloads
- Lag features and rolling averages
- Random Forest regression baseline
- Chronological holdout evaluation using MAE and RMSE
- Streamlit dashboard for visualising a recent speed series and next-step estimate
- CSV input support for real datasets with `timestamp` and `speed_kmh` columns
- Unit tests for feature engineering and validation

## Quick start
Python 3.10+ is recommended.

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
python train.py
pytest -q
streamlit run app/dashboard.py
```

The first training run creates `models/traffic_model.joblib` locally. Model files are ignored by Git.

## Train with your own dataset
Prepare a CSV containing:
- `timestamp`: date/time for each observation
- `speed_kmh`: observed speed in kilometres per hour

```bash
python train.py --data path/to/traffic.csv
```

Document the dataset's source, licence, location coverage, units, missing-value handling and preprocessing before reporting results. Do not upload restricted data.

## Project structure
```text
.
├── app/dashboard.py
├── data/README.md
├── reports/model_evaluation.md
├── src/traffic_model.py
├── tests/test_traffic_model.py
├── train.py
├── requirements.txt
└── .gitignore
```

## Evaluation
The training script uses a chronological split: earlier observations train the model and later observations are held out. It reports MAE and RMSE in km/h. Before making performance claims, compare against a persistence baseline (predict the next speed as the latest observed speed) and evaluate on a properly licensed, held-out real dataset.

## Limitations and next steps
- This is a one-step baseline, not a graph neural network or multi-horizon forecasting model.
- The dashboard currently displays synthetic data for a safe, reproducible demo.
- No live traffic API or real-time alerts are implemented.
- Future work: public sensor dataset integration, persistence baseline comparison, multiple forecast horizons, spatial graph features, model tracking, Docker deployment and CI.

## Resume guidance
Only describe features you have run and verified. Do not claim live traffic prediction or real-world accuracy based on synthetic data.
