# Data

The first-run demo generates synthetic traffic-speed observations. These are illustrative values, not measurements from a real city or road network.

## Use a real dataset
Prepare a CSV with:
- `timestamp`: observation timestamp
- `speed_kmh`: observed speed in kilometres per hour

Run `python train.py --data path/to/traffic.csv`.

Before publishing results, document the dataset source, licence, geographic coverage, units, missing-value handling and preprocessing. Do not commit restricted or personally identifying data.
