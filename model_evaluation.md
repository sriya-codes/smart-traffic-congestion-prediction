# Model evaluation protocol

The training script uses a chronological holdout: earlier observations train the model and later observations are reserved for testing. It reports MAE and RMSE in km/h.

Before making performance claims:
- Run the tests and training script.
- Record the dataset source and licence.
- Compare against a persistence baseline (next speed equals the latest observed speed).
- Report results on a held-out real dataset, not only generated demo data.
- Check performance by time of day and forecast horizon.
- Describe this as a baseline, not a graph neural network or live traffic system.
