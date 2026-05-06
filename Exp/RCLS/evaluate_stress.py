from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rcls_metrics import aggregate_daily_metrics


STRESS_FEATURES = {
    "all": None,
    "high_volatility": "ctx_realized_volatility",
    "high_dispersion": "ctx_dispersion",
    "high_synchronism": "ctx_synchronism",
    "high_downside": "ctx_downside_volatility",
    "high_pca_ratio": "ctx_pca_ratio",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    return parser.parse_args()


def subset_days(day_df, subset, feature):
    if subset == "all" or feature is None or feature not in day_df:
        return set(day_df["day_idx"])
    threshold = day_df[feature].quantile(0.70)
    return set(day_df.loc[day_df[feature] >= threshold, "day_idx"])


def main():
    args = parse_args()
    rows = []
    for path in sorted(args.results_root.glob("preds_*.csv")):
        df = pd.read_csv(path)
        df = df[df["split"] == "test"].copy()
        if df.empty:
            continue
        day_df = df.drop_duplicates("day_idx")
        for subset, feature in STRESS_FEATURES.items():
            days = subset_days(day_df, subset, feature)
            sub = df[df["day_idx"].isin(days)]
            metrics, _ = aggregate_daily_metrics(sub)
            row = {
                "dataset": df["dataset"].iloc[0],
                "model": df["model"].iloc[0],
                "seed": int(df["seed"].iloc[0]),
                "subset": subset,
            }
            row.update(metrics)
            rows.append(row)
    pd.DataFrame(rows).to_csv(args.results_root / "summary_stress.csv", index=False)
    print(f"Wrote {args.results_root / 'summary_stress.csv'}")


if __name__ == "__main__":
    main()
