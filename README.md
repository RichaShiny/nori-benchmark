# Nori V1 Benchmark

Benchmarking [Synthefy's Nori V1](https://github.com/Synthefy/synthefy-nori), an open-weight
tabular foundation model, against sklearn baselines. Nori predicts via in-context learning:
no training, no feature engineering, no hyperparameter tuning, one forward pass.

## Results

All models untuned, 75/25 split, seed 42. Nori runs on CPU (MacBook Air).

### Diabetes (442 rows, 10 features)

| Model | R2 | MAE | Predict time |
|---|---|---|---|
| **Nori V1 (30M)** | **0.5351** | **40.25** | 20.7s |
| RandomForest (300 trees) | 0.4741 | 44.20 | <0.1s |
| Ridge | 0.4384 | 46.14 | <0.1s |
| HistGradientBoosting | 0.3928 | 48.53 | <0.1s |

### California housing, low-data regime (3,000 train rows)

| Model | R2 | MAE | Predict time |
|---|---|---|---|
| **Nori V1 (30M)** | **0.8210** | **0.297** | 185.4s |
| HistGradientBoosting | 0.7787 | 0.350 | <0.1s |
| RandomForest (300 trees) | 0.7340 | 0.383 | <0.1s |
| Ridge | 0.4729 | 0.521 | <0.1s |

### California housing, full size (15,000 train rows)

| Model | R2 | MAE |
|---|---|---|
| HistGradientBoosting | 0.8406 | 0.309 |
| RandomForest (300 trees) | 0.8168 | 0.320 |
| Ridge | 0.6093 | 0.530 |
| Nori V1 | not run (CPU-impractical at this size; see limitations) | |

### Takeaways

- **Data efficiency is the headline:** Nori with 3,000 training rows (0.821) nearly
  matches HistGradientBoosting with 15,000 rows (0.841). In-context learning bought
  the equivalent of roughly 5x more data, with zero training or tuning.
- **Small tables: Nori dominates.** On diabetes it beat the best baseline by +13%
  relative R2.
- **At scale, gradient boosting still wins**, consistent with Synthefy's own
  documentation that Nori targets small-to-mid tables.
- **The tradeoff is inference cost:** ~185s on CPU at 3k context rows vs sub-second
  for the sklearn models.

## Limitations found

- **No Apple Silicon GPU support:** Nori v0.18 fails on `device="mps"` because the
  checkpoint contains float64 buffers, which Metal cannot represent. The failure is
  inside the package's own `model.to(device)` call, so it is not fixable from user
  code. CPU-only on Mac.
- The marketing quickstart shows `predict(X_test, X_train, y_train)`; the actual
  API (v0.17+) is sklearn-style `fit`/`predict`.
- Always pass a model size (`"nori-30m"` or `"nori-6m"`); omitting it silently
  falls back to the weaker 6M base. The "thinking" variant is hosted-API only.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py                          # all datasets
python main.py --datasets diabetes      # single dataset
python main.py --max-rows 4000          # cap training rows (CPU-friendly)
python main.py --model nori-6m          # smaller/faster checkpoint
python main.py --skip-nori              # baselines only
```

First Nori run downloads the checkpoint (~117MB) from Hugging Face and caches it.
Results save to `results/results.csv` and `results/r2_by_model.csv`.

## Add your own dataset

Drop a CSV in `data/` and register it in `src/nori_bench/datasets.py`:

```python
CSV_DATASETS = {
    "my_dataset": ("my_dataset.csv", "target_column_name"),
}
```

## Structure

```
main.py                     CLI entry point
src/nori_bench/
  datasets.py               dataset registry
  models.py                 model registry (baselines + Nori)
  runner.py                 benchmark loop and metrics
```

## Notes

- Nori is regression only; NaNs in features are handled natively, no imputation needed
- Nori itself is Apache 2.0 (Synthefy); this benchmark code is MIT
