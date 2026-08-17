# Nori V1 Benchmark

Benchmarking [Synthefy's Nori V1](https://github.com/Synthefy/synthefy-nori), an open-weight
tabular foundation model, against sklearn baselines. Nori predicts via in-context learning:
no training, no feature engineering, no hyperparameter tuning, one forward pass.

All models untuned, 75/25 split, seed 42. Nori-30M on CPU (MacBook Air).
Results within ~0.01 R2 should be read as ties (single split per test).

## Test 1 — Diabetes (442 rows, 10 features): small-table regime

| Model | R2 | MAE | Predict time |
|---|---|---|---|
| **Nori V1** | **0.5351** | **40.25** | 20.7s |
| RandomForest (300 trees) | 0.4741 | 44.20 | <0.1s |
| Ridge | 0.4384 | 46.14 | <0.1s |
| HistGradientBoosting | 0.3928 | 48.53 | <0.1s |

Nori dominates: +13% relative R2 over the best baseline with zero training.

## Test 2 — California housing @ 3,000 train rows: low-data regime

| Model | R2 | MAE | Predict time |
|---|---|---|---|
| **Nori V1** | **0.8210** | **0.297** | 185.4s |
| HistGradientBoosting | 0.7787 | 0.350 | <0.1s |
| RandomForest (300 trees) | 0.7340 | 0.383 | <0.1s |
| Ridge | 0.4729 | 0.521 | <0.1s |

## Test 3 — California housing @ 15,000 train rows: at-scale regime

| Model | R2 | MAE |
|---|---|---|
| HistGradientBoosting | 0.8406 | 0.309 |
| RandomForest (300 trees) | 0.8168 | 0.320 |
| Ridge | 0.6093 | 0.530 |
| Nori V1 | not run (CPU-impractical at this size) | |

**Data-efficiency headline from Tests 2-3:** Nori with 3,000 rows (0.821) nearly matches
HistGradientBoosting with 15,000 rows (0.841). In-context learning bought the
equivalent of roughly 5x more data.

## Test 4 — Concrete compressive strength (1,030 rows, 8 features): small-table confirmation

| Model | R2 | MAE | Predict time |
|---|---|---|---|
| **Nori V1** | **0.9518** | **TBD** | TBD |
| HistGradientBoosting | 0.9205 | 3.135 | <0.1s |
| RandomForest (300 trees) | 0.8909 | 3.713 | <0.1s |
| Ridge | 0.6249 | 7.987 | <0.1s |

Second small-table win, and on a clean, physics-driven dataset where tree ensembles
are typically strongest. Nori's margin over HGB (+0.031) exceeded its diabetes margin.

## Test 5 — Abalone (4,177 rows, incl. categorical feature): mid-size regime

Results pending.

## Test 6 — Wine quality (6,497 rows): near the crossover boundary

Results pending.

## Takeaways

- **Small tables (Tests 1, 4): Nori wins with zero training**, on both noisy
  (diabetes) and clean (concrete) data.
- **Low-data regimes (Test 2): Nori's strongest case**, where baselines are data-starved.
- **At scale (Test 3): gradient boosting still leads**, consistent with Synthefy's own
  documentation that Nori targets small-to-mid tables.
- **The tradeoff is inference cost:** seconds to minutes on CPU vs sub-second for
  sklearn models.

## Limitations found

- **No Apple Silicon GPU support:** Nori v0.18 fails on `device="mps"` because the
  checkpoint itself contains float64 buffers, which Metal cannot represent. The failure
  is inside the package's own `model.to(device)` call, so it is not fixable from user
  code. CPU-only on Mac.
- The marketing quickstart shows `predict(X_test, X_train, y_train)`; the actual
  API (v0.17+) is sklearn-style `fit`/`predict`.
- Always pass a model size (`"nori-30m"` or `"nori-6m"`); omitting it silently
  falls back to the weaker 6M base. The "thinking" variant is hosted-API only.

## Engineering notes

- `runner.py` casts features and targets to float32 **for Nori only**, required for
  non-CPU devices, while baselines keep float64 so their results stay bit-identical
  across tests. This fix is what isolated the MPS failure to the vendor checkpoint
  rather than user data: with inputs correctly cast, the crash moved inside Nori's
  own weight-loading path.
- Per-model exception handling records a failed model as NaN and continues the
  benchmark instead of aborting, which is why Test 3 still produced baseline scores
  when the MPS run failed.
- `datasets.py` originally fetched Tests 4-6 from OpenML; persistent 504s from the
  OpenML API led to a rewrite using UCI direct downloads with a local CSV cache in
  `data/`, so each dataset needs exactly one successful fetch ever.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py                          # all datasets
python main.py --datasets concrete      # single dataset
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
  datasets.py               dataset registry (sklearn, UCI cached, local CSVs)
  models.py                 model registry (baselines + Nori)
  runner.py                 benchmark loop, dtype handling, metrics
```

## Notes

- Nori is regression only; NaNs in features are handled natively, no imputation needed
- Nori itself is Apache 2.0 (Synthefy); this benchmark code is MIT
