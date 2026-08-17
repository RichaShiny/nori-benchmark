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

| Model | R2 | MAE | Predict time |
|---|---|---|---|
| **Nori V1** | **0.6000** | **1.439** | 172.9s |
| RandomForest (300 trees) | 0.5367 | 1.551 | <0.1s |
| HistGradientBoosting | 0.5336 | 1.527 | <0.1s |
| Ridge | 0.5303 | 1.609 | <0.1s |

The baselines were statistically tied (0.530-0.537), a classic noise-floor profile
where even linear regression matches tree ensembles. Nori broke through it by +0.063
R2 (+12% relative), its largest margin yet, on a dataset with an encoded categorical
feature.

## Test 6 — Wine quality (6,497 rows, 12 features): near the crossover boundary

| Model | R2 | MAE | Predict time |
|---|---|---|---|
| **Nori V1** | **0.5427** | **0.390** | 467.8s |
| RandomForest (300 trees) | 0.4917 | 0.440 | <0.1s |
| HistGradientBoosting | 0.4412 | 0.493 | <0.1s |
| Ridge | 0.2613 | 0.571 | <0.1s |

Largest context size tested (4,872 rows) and another noisy target (human taste
ratings). Ridge collapsing to 0.26 while trees held ~0.44-0.49 shows real nonlinear
signal; Nori beat the tree ensembles anyway, +0.051 over the best.

## Test 7 — Airfoil self-noise (1,503 rows, 5 features): strongly nonlinear physics data

| Model | R2 | MAE | Predict time |
|---|---|---|---|
| **Nori V1** | **0.9842** | **0.556** | 33.8s |
| RandomForest (300 trees) | 0.9318 | 1.319 | <0.1s |
| HistGradientBoosting | 0.9262 | 1.334 | <0.1s |
| Ridge | 0.4845 | 3.926 | <0.1s |

The most decisive result in the suite. Ridge collapsing to 0.48 shows strongly
nonlinear signal; trees handled it well (0.93), and Nori still eliminated 77% of the
best baseline's residual error (unexplained variance 6.8% down to 1.6%), with less
than half the MAE.

## Test 8 — Auto MPG (398 rows, native missing values): NaN handling

| Model | R2 | MAE | Predict time |
|---|---|---|---|
| **Nori V1** | **0.9181** | **1.589** | 8.4s |
| HistGradientBoosting | 0.8903 | TBD | <0.1s |
| RandomForest (300 trees) | 0.8882 | TBD | <0.1s |
| Ridge | failed (cannot accept NaN input) | | |

NaNs passed straight through with no imputation. Nori and both tree models handled
them natively (modern sklearn RandomForest accepts NaNs); Ridge failed as expected,
recorded as NaN by the runner's per-model exception handling.

## Takeaways

Across eight tests, Nori won every contested matchup (Tests 1, 2, 4-8), from ~300
to 4,872 context rows:

- **The noisier the data, the bigger Nori's edge.** Its two largest margins came on
  noisy targets: abalone (+0.063 over a baseline noise floor where Ridge matched
  RandomForest) and diabetes (+13% relative). On wine, another noisy target with
  clearly nonlinear signal, it beat tree ensembles by +0.051.
- **Clean data is no refuge for baselines either:** on concrete (0.952 vs 0.921)
  and especially airfoil, where against trees already at 0.93 on strongly nonlinear
  physics data, Nori hit 0.984, eliminating 77% of the residual error.
- **Native NaN handling works as advertised** (Test 8): missing values passed
  through with no imputation, and Nori still won.
- **Data efficiency (Tests 2-3):** Nori with 3,000 rows (0.821) nearly matches
  HistGradientBoosting with 15,000 rows (0.841), roughly a 5x data equivalent.
- **At scale, gradient boosting presumably retakes the lead** (Test 3's 15k-row run
  was CPU-impractical for Nori; Synthefy's own docs concede large tables to boosting).
- **The tradeoff is inference cost, and it scales steeply:** 21s at 331 context rows,
  173s at 3.1k, 468s at 4.9k, vs sub-second for every sklearn model.

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
