from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rcls_metrics import aggregate_daily_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    return parser.parse_args()


def prediction_files(root):
    return sorted(root.glob("preds_*.csv"))


def main():
    args = parse_args()
    rows = []
    for path in prediction_files(args.results_root):
        df = pd.read_csv(path)
        if "split" in df:
            df = df[df["split"] == "test"].copy()
        if df.empty:
            continue
        metrics, _ = aggregate_daily_metrics(df)
        row = {
            "dataset": df["dataset"].iloc[0],
            "model": df["model"].iloc[0],
            "seed": int(df["seed"].iloc[0]),
        }
        row.update(metrics)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["dataset", "model", "seed"]) if rows else pd.DataFrame()
    out.to_csv(args.results_root / "summary_main.csv", index=False)
    out.to_csv(args.results_root / "summary_ablation.csv", index=False)
    if not out.empty:
        metric_cols = [c for c in out.columns if c not in {"dataset", "model", "seed"}]
        mean_rows = []
        for (dataset, model), group in out.groupby(["dataset", "model"]):
            row = {"dataset": dataset, "model": model, "num_seeds": group["seed"].nunique()}
            for col in metric_cols:
                row[f"{col}_mean"] = group[col].mean()
                row[f"{col}_std"] = group[col].std()
            mean_rows.append(row)
        pd.DataFrame(mean_rows).to_csv(args.results_root / "summary_main_mean_std.csv", index=False)
    pd.DataFrame(
        [
            {"reference": "stockmixer_seed1_100e", "dataset": "NASDAQ", "ic": 0.0366, "icir": 0.388, "p10": 0.527, "sharpe_legacy": 1.540},
            {"reference": "cagclean_seed1_100e", "dataset": "NASDAQ", "ic": 0.03255, "icir": 0.41803, "p10": 0.51857, "sharpe_legacy": 0.68693},
        ]
    ).to_csv(args.results_root / "summary_paper_reference.csv", index=False)
    print(out.to_string(index=False) if not out.empty else "No prediction files found.")


if __name__ == "__main__":
    main()
