"""Dataset registry. Bundled sklearn sets, UCI direct downloads, local CSVs."""
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# name -> (csv filename, target column)
CSV_DATASETS = {
    # "my_dataset": ("my_dataset.csv", "target"),
}

UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases"


def _cached(fname: str, loader):
    """Download once into data/, reuse thereafter."""
    path = DATA_DIR / fname
    if path.exists():
        return pd.read_csv(path)
    df = loader()
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _load_abalone():
    cols = ["sex", "length", "diameter", "height", "whole_weight",
            "shucked_weight", "viscera_weight", "shell_weight", "rings"]
    df = _cached("abalone.csv",
                 lambda: pd.read_csv(f"{UCI}/abalone/abalone.data", names=cols))
    df["sex"] = df["sex"].astype("category").cat.codes
    return df.drop(columns=["rings"]), df["rings"].astype(float)


def _load_wine():
    def fetch():
        red = pd.read_csv(f"{UCI}/wine-quality/winequality-red.csv", sep=";")
        white = pd.read_csv(f"{UCI}/wine-quality/winequality-white.csv", sep=";")
        red["is_red"] = 1
        white["is_red"] = 0
        return pd.concat([red, white], ignore_index=True)
    df = _cached("wine.csv", fetch)
    return df.drop(columns=["quality"]), df["quality"].astype(float)


def _load_concrete():
    def fetch():
        df = pd.read_excel(f"{UCI}/concrete/compressive/Concrete_Data.xls")
        df.columns = ["cement", "slag", "fly_ash", "water", "superplasticizer",
                      "coarse_agg", "fine_agg", "age", "strength"]
        return df
    df = _cached("concrete.csv", fetch)
    return df.drop(columns=["strength"]), df["strength"]


REMOTE_DATASETS = {
    "concrete": _load_concrete,   # 1,030 rows
    "abalone": _load_abalone,     # 4,177 rows
    "wine": _load_wine,           # 6,497 rows
}


def load_dataset(name: str):
    if name == "diabetes":
        data = load_diabetes(as_frame=True)
        return data.data, data.target
    if name == "california":
        data = fetch_california_housing(as_frame=True)
        return data.data, data.target
    if name in REMOTE_DATASETS:
        return REMOTE_DATASETS[name]()
    if name in CSV_DATASETS:
        fname, target = CSV_DATASETS[name]
        df = pd.read_csv(DATA_DIR / fname)
        return df.drop(columns=[target]), df[target]
    raise ValueError(f"Unknown dataset: {name}. Available: {available()}")


def available() -> list[str]:
    return ["diabetes", "california", *REMOTE_DATASETS, *CSV_DATASETS]
