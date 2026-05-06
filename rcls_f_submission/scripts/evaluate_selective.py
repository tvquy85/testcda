import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from summarize_results import compute_metrics


RELIABILITY_COLUMNS = [
    "dataset",
    "model",
    "seed",
    "confidence_source",
    "selection_unit",
    "coverage",
    "num_days",
    "num_rows",
    "ic",
    "rankic",
    "precision_at_10",
    "precision_at_20",
    "mae",
]


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=root)
    parser.add_argument("--results-dir", "--input-dir", dest="results_dir", type=Path, default=None)
    return parser.parse_args()


def get_results_dir(args):
    if args.results_dir is not None:
        return args.results_dir
    return args.output_root / "results"


def gate_entropy(frame, regime_cols):
    probs = frame[regime_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    values = probs.to_numpy(dtype=float)
    row_sum = values.sum(axis=1, keepdims=True)
    values = np.divide(values, row_sum, out=np.zeros_like(values), where=row_sum > 0)
    values = np.clip(values, 1e-12, 1.0)
    return -(values * np.log(values)).sum(axis=1)


def add_confidence(df):
    if "sigma" in df.columns and pd.to_numeric(df["sigma"], errors="coerce").notna().any():
        df["confidence"] = -pd.to_numeric(df["sigma"], errors="coerce")
        return df, "sigma"

    regime_cols = [c for c in ["regime_0", "regime_1", "regime_2"] if c in df.columns]
    active_cols = []
    for col in regime_cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            df[col] = numeric
            active_cols.append(col)
    if active_cols:
        df["confidence"] = -gate_entropy(df, active_cols)
        return df, "gate_entropy"

    df["confidence"] = pd.to_numeric(df["pred"], errors="coerce").abs()
    return df, "abs_pred"


def select_stock_rows_by_coverage(df, coverage):
    rows = []
    for _, day_df in df.groupby("day_idx"):
        day_df = day_df.sort_values("confidence", ascending=False)
        keep = max(1, int(np.ceil(len(day_df) * coverage)))
        rows.append(day_df.head(keep))
    if not rows:
        return df.iloc[0:0].copy()
    return pd.concat(rows, ignore_index=True)


def select_days_by_coverage(df, coverage):
    day_conf = df[["day_idx", "confidence"]].drop_duplicates("day_idx")
    if day_conf.empty:
        return df.iloc[0:0].copy()
    keep = max(1, int(np.ceil(len(day_conf) * coverage)))
    selected_days = set(
        day_conf.sort_values("confidence", ascending=False).head(keep)["day_idx"]
    )
    return df[df["day_idx"].isin(selected_days)].copy()


def summarize_file(path):
    df = pd.read_csv(path)
    if "split" in df.columns:
        df = df[df["split"] == "test"]
    if df.empty:
        return []
    dataset = str(df["dataset"].iloc[0])
    model = str(df["model"].iloc[0])
    seed = int(df["seed"].iloc[0])
    df, confidence_source = add_confidence(df)
    selection_unit = "day" if confidence_source == "gate_entropy" else "stock_row"

    rows = []
    for coverage in [1.0, 0.7, 0.5, 0.3]:
        if selection_unit == "day":
            selected = select_days_by_coverage(df, coverage)
        else:
            selected = select_stock_rows_by_coverage(df, coverage)
        metrics = compute_metrics(selected)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "confidence_source": confidence_source,
                "selection_unit": selection_unit,
                "coverage": coverage,
                "num_days": metrics["num_days"],
                "num_rows": metrics["num_rows"],
                "ic": metrics["ic"],
                "rankic": metrics["rankic"],
                "precision_at_10": metrics["precision_at_10"],
                "precision_at_20": metrics["precision_at_20"],
                "mae": metrics["mae"],
            }
        )
    return rows


def main():
    args = parse_args()
    results_dir = get_results_dir(args)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(results_dir.glob("preds_*.csv")):
        rows.extend(summarize_file(path))
    output = results_dir / "summary_reliability.csv"
    pd.DataFrame(rows, columns=RELIABILITY_COLUMNS).to_csv(output, index=False)
    print("Wrote {}".format(output))


if __name__ == "__main__":
    main()
