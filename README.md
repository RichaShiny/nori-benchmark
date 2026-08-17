# Nori V1 Benchmark

Benchmarks Synthefy's [Nori V1](https://github.com/Synthefy/synthefy-nori) tabular
foundation model against tuned-free sklearn baselines (Ridge, RandomForest,
HistGradientBoosting) on regression datasets. Nori predicts via in-context
learning: no training, no hyperparameter tuning, one forward pass.

## Setup (VS Code)

1. Open this folder in VS Code (File > Open Folder)
2. Create the environment in the built-in terminal (Ctrl+`):

```bash
python -m venv .venv
# Mac/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# CPU-only Linux/Windows: install CPU torch first (skips the 3GB CUDA bundle)
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

3. Select the interpreter: Ctrl+Shift+P > "Python: Select Interpreter" > `.venv`

## Run

```bash
python main.py                        # all datasets, CPU
python main.py --datasets diabetes    # single dataset
python main.py --device cuda          # if you have a GPU
python main.py --model nori-6m        # smaller/faster checkpoint
python main.py --skip-nori            # baselines only, no weight download
```

First Nori run downloads the checkpoint from Hugging Face and caches it.
NaNs in features are handled natively, no imputation needed.
Results land in `results/results.csv` and `results/r2_by_model.csv`.

## Add your own data

Drop a CSV in `data/` and register it in `src/nori_bench/datasets.py`:

```python
CSV_DATASETS = {
    "my_dataset": ("my_dataset.csv", "target_column_name"),
}
```

## API gotcha

Synthefy's marketing page shows `model.predict(X_test, X_train, y_train)`.
The actual API (v0.17.x) is sklearn-style:

```python
model = NoriRegressor(model="nori-30m", device="cpu")
model.fit(X_train, y_train)   # stores context only, no training
pred = model.predict(X_test)
```

Always name a size ("nori-30m" or "nori-6m"). Omitting `model=` silently
falls back to the weaker 6M base. The "thinking" variant is hosted-API only
and raises an error if you select it locally.

Useful extras: `output_type`/`quantiles` in `predict()` for uncertainty,
`text_columns` in the constructor for text features (MiniLM embedder),
`model_path` to load a local checkpoint.

## Notes

- Nori is regression only (`NoriRegressor`); there is no classifier class
- Synthefy's own blog says gradient boosting still wins past ~100k cells,
  hence the `--max-rows` subsample cap (default 20k)
- Weights and code are Apache 2.0, free for commercial use

## Structure

```
nori-benchmark/
  main.py                     CLI entry point
  requirements.txt
  data/                       your CSVs go here
  results/                    generated output
  src/nori_bench/
    datasets.py               dataset registry
    models.py                 model registry
    runner.py                 benchmark loop and metrics
```
