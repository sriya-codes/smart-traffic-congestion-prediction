"""Train a traffic-speed model from CSV data or synthetic demo data."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from src.traffic_model import generate_demo_data, train_baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="CSV with timestamp,speed_kmh columns")
    parser.add_argument("--output", type=Path, default=Path("models/traffic_model.joblib"))
    args = parser.parse_args()
    if args.data:
        data = pd.read_csv(args.data)
        print(f"Training on supplied dataset: {args.data}")
    else:
        data = generate_demo_data()
        print("Using SYNTHETIC demo data; these are not real traffic observations.")
    model, mae, rmse, rows = train_baseline(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    print(f"Saved model: {args.output}")
    print(f"Chronological test rows: {rows}")
    print(f"MAE: {mae:.3f} km/h | RMSE: {rmse:.3f} km/h")
    print("Metrics are dataset-specific; compare against a persistence baseline before claiming performance.")


if __name__ == "__main__":
    main()
