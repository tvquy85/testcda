from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rcls_metrics import aggregate_daily_metrics, masked_mae


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for path in sorted(args.results_root.glob("preds_*.csv")):
        df = pd.read_csv(path)
        df = df[df["split"] == "test"].copy()
        if df.empty:
            continue
        day = df.drop_duplicates("day_idx").copy()
        if day["gate_entropy"].std() > 1e-12:
            source = "gate_entropy"
            day = day.sort_values("gate_entropy", ascending=True)
            selection_unit = "day"
        else:
            source = "abs_pred"
            df["_conf"] = df["pred"].abs()
            selection_unit = "row"
        for coverage in [1.0, 0.7, 0.5, 0.3]:
            if selection_unit == "day":
                keep_days = set(day.head(max(1, int(len(day) * coverage)))["day_idx"])
                sub = df[df["day_idx"].isin(keep_days)]
            else:
                sub = df.sort_values("_conf", ascending=False).head(max(1, int(len(df) * coverage)))
            metrics, _ = aggregate_daily_metrics(sub)
            row = {
                "dataset": df["dataset"].iloc[0],
                "model": df["model"].iloc[0],
                "seed": int(df["seed"].iloc[0]),
                "confidence_source": source,
                "selection_unit": selection_unit,
                "coverage": coverage,
                "num_days": sub["day_idx"].nunique(),
                "num_rows": len(sub),
                "mae": masked_mae(sub["pred"], sub["target"], sub["mask"]),
                "nll": float("nan"),
            }
            row.update(metrics)
            rows.append(row)
    pd.DataFrame(rows).to_csv(args.results_root / "summary_reliability.csv", index=False)
    print(f"Wrote {args.results_root / 'summary_reliability.csv'}")


if __name__ == "__main__":
    main()
