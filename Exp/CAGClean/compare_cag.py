from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["mse", "IC", "RIC", "prec_10", "sharpe5"]


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Compare clean StockMixer and CAG-MLP runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", type=Path, default=repo_root / "Exp" / "results" / "cag_context_matched60")
    return parser.parse_args()


def main():
    args = parse_args()
    metadata_path = args.results_dir / "run_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    df = pd.read_csv(metadata_path)
    keys = ["model", "market", "numpy_seed", "torch_seed", "epochs"]
    df = df.sort_values("run_timestamp").drop_duplicates(keys, keep="last")
    rows = []
    for _, row in df.iterrows():
        out = {
            "model": row["model"],
            "market": row["market"],
            "epochs": row["epochs"],
            "best_epoch": row["best_epoch"],
            "param_count": row["param_count"],
        }
        for metric in METRICS:
            out[metric] = row[f"best_test_{metric}"]
        rows.append(out)

    summary = pd.DataFrame(rows)
    if {"stockmixer", "cag_gated_context"}.issubset(set(summary["model"])):
        base = summary.loc[summary["model"] == "stockmixer"].iloc[-1]
        cag = summary.loc[summary["model"] == "cag_gated_context"].iloc[-1]
        delta = {
            "model": "delta_cag_minus_stockmixer",
            "market": cag["market"],
            "epochs": cag["epochs"],
            "best_epoch": np.nan,
            "param_count": cag["param_count"] - base["param_count"],
        }
        for metric in METRICS:
            delta[metric] = cag[metric] - base[metric]
        summary = pd.concat([summary, pd.DataFrame([delta])], ignore_index=True)

    ref_rows = []
    for _, row in df.iterrows():
        if row["market"] != "NASDAQ":
            continue
        if "paper_cag_IC" not in row:
            continue
        ref_rows.extend(
            [
                {
                    "reference": "paper_stockmixer",
                    "IC": row.get("paper_stockmixer_IC", np.nan),
                    "RIC": row.get("paper_stockmixer_RIC", np.nan),
                    "prec_10": row.get("paper_stockmixer_prec_10", np.nan),
                    "sharpe5": row.get("paper_stockmixer_sharpe5", np.nan),
                },
                {
                    "reference": "paper_cag",
                    "IC": row.get("paper_cag_IC", np.nan),
                    "RIC": row.get("paper_cag_RIC", np.nan),
                    "prec_10": row.get("paper_cag_prec_10", np.nan),
                    "sharpe5": row.get("paper_cag_sharpe5", np.nan),
                },
            ]
        )
        break
    if ref_rows:
        pd.DataFrame(ref_rows).to_csv(args.results_dir / "summary_paper_reference.csv", index=False)

    output_path = args.results_dir / "summary_cag_comparison.csv"
    summary.to_csv(output_path, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
