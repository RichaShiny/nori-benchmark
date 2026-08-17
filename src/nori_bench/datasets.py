"""Dataset registry for the benchmark.

Bundled sklearn datasets require no network. Add your own CSVs to data/
and register them in CSV_DATASETS below.
"""
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# name -> (csv filename, target column)
CSV_DATASETS = {
    # "my_dataset": ("my_dataset.csv", "target"),
}


def load_dataset(name: str):
    """Return (X, y) as a DataFrame and Series for a registered dataset."""
    if name == "diabetes":
        data = load_diabetes(as_frame=True)
        return data.data, data.target
    if name == "california":
        data = fetch_california_housing(as_frame=True)
        return data.data, data.target
    if name in CSV_DATASETS:
        fname, target = CSV_DATASETS[name]
        df = pd.read_csv(DATA_DIR / fname)
        return df.drop(columns=[target]), df[target]
    raise ValueError(f"Unknown dataset: {name}. Available: {available()}")


def available() -> list[str]:
    return ["diabetes", "california", *CSV_DATASETS]
