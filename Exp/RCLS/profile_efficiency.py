from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    meta = args.results_root / "run_metadata.csv"
    if not meta.exists():
        pd.DataFrame().to_csv(args.results_root / "summary_efficiency.csv", index=False)
        return
    df = pd.read_csv(meta)
    cols = [
        "dataset",
        "model",
        "seed",
        "num_params",
        "train_time_sec",
        "total_time_sec",
        "max_vram_gb",
    ]
    out = df[[c for c in cols if c in df]].copy()
    if "num_params" in out:
        out["model_size_mb"] = out["num_params"] * 4 / (1024 ** 2)
    out.to_csv(args.results_root / "summary_efficiency.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
