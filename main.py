"""CLI entry point.

Examples:
    python main.py                          # all datasets, CPU
    python main.py --datasets diabetes      # one dataset
    python main.py --device cuda            # GPU
    python main.py --skip-nori              # baselines only (no weight download)
"""
import argparse
from pathlib import Path

from src.nori_bench.datasets import available
from src.nori_bench.runner import run_benchmark, summarize

RESULTS_DIR = Path(__file__).parent / "results"


def main():
    parser = argparse.ArgumentParser(description="Nori V1 vs sklearn baselines")
    parser.add_argument("--datasets", nargs="+", default=available(),
                        choices=available(), help="Datasets to run")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                        help="Device for Nori inference")
    parser.add_argument("--skip-nori", action="store_true",
                        help="Run baselines only")
    parser.add_argument("--model", default="nori-30m",
                        choices=["nori-30m", "nori-6m"],
                        help="Nori checkpoint (thinking variant is hosted-API only)")
    parser.add_argument("--max-rows", type=int, default=20_000,
                        help="Subsample cap per dataset (Nori is built for small-to-mid tables)")
    args = parser.parse_args()

    results = run_benchmark(
        args.datasets, device=args.device,
        skip_nori=args.skip_nori, max_rows=args.max_rows,
        nori_model=args.model,
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    results.to_csv(RESULTS_DIR / "results.csv", index=False)

    pivot = summarize(results)
    pivot.to_csv(RESULTS_DIR / "r2_by_model.csv")
    print("\n=== R2 by model ===")
    print(pivot.to_string())
    print(f"\nSaved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
