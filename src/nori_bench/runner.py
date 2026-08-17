"""Benchmark loop: run every model on every dataset, collect metrics."""
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

from .datasets import load_dataset
from .models import get_models


def run_benchmark(
    dataset_names: list[str],
    device: str = "cpu",
    skip_nori: bool = False,
    nori_model: str = "nori-30m",
    test_size: float = 0.25,
    max_rows: int | None = 20_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run all models on all datasets. Returns a tidy results DataFrame."""
    rows = []
    for ds_name in dataset_names:
        X, y = load_dataset(ds_name)
        if max_rows and len(X) > max_rows:
            X = X.sample(max_rows, random_state=seed)
            y = y.loc[X.index]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )
        print(f"\n[{ds_name}] train={X_train.shape} test={X_test.shape}")

        for model_name, model in get_models(
            device=device, skip_nori=skip_nori, nori_model=nori_model
        ).items():
            try:
                # MPS doesn't support float64; feed Nori float32
                if model_name == "NoriV1":
                    Xtr = X_train.astype("float32")
                    Xte = X_test.astype("float32")
                    ytr = y_train.astype("float32")
                else:
                    Xtr, Xte, ytr = X_train, X_test, y_train

                t0 = time.time()
                model.fit(Xtr, ytr)
                fit_s = time.time() - t0

                t0 = time.time()
                pred = np.asarray(model.predict(Xte)).ravel()
                pred_s = time.time() - t0

                rows.append({
                    "dataset": ds_name,
                    "model": model_name,
                    "r2": r2_score(y_test, pred),
                    "mae": mean_absolute_error(y_test, pred),
                    "rmse": root_mean_squared_error(y_test, pred),
                    "fit_s": round(fit_s, 2),
                    "predict_s": round(pred_s, 2),
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                })
                print(f"  {model_name:<22} R2={rows[-1]['r2']:.4f}  "
                      f"MAE={rows[-1]['mae']:.3f}  "
                      f"({fit_s:.1f}s fit / {pred_s:.1f}s predict)")
            except Exception:
                print(f"  {model_name:<22} FAILED")
                traceback.print_exc()
                rows.append({
                    "dataset": ds_name, "model": model_name, "r2": np.nan,
                    "mae": np.nan, "rmse": np.nan, "fit_s": np.nan,
                    "predict_s": np.nan, "n_train": len(X_train),
                    "n_test": len(X_test),
                })
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Pivot to a dataset x model R2 table for quick comparison."""
    return results.pivot(index="dataset", columns="model", values="r2").round(4)