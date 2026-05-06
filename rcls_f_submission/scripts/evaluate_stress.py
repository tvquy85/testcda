import argparse
from pathlib import Path

import pandas as pd

from summarize_results import compute_metrics


STRESS_COLUMNS = [
    "dataset",
    "model",
    "regime_mode",
    "seed",
    "stress_source",
    "subset",
    "num_days",
    "num_rows",
    "ic",
    "rankic",
    "precision_at_10",
    "precision_at_20",
    "long_short_return",
    "sharpe",
]


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=root)
    parser.add_argument("--results-dir", "--input-dir", dest="results_dir", type=Path, default=None)
    parser.add_argument("--top-frac", type=float, default=0.30)
    return parser.parse_args()


def get_results_dir(args):
    if args.results_dir is not None:
        return args.results_dir
    return args.output_root / "results"


def frame_regime_mode(df):
    if "regime_mode" in df.columns and df["regime_mode"].notna().any():
        value = str(df["regime_mode"].dropna().iloc[0])
        return value if value else "legacy_delta"
    return "legacy_delta"


def top_days(day_df, column, frac):
    values = pd.to_numeric(day_df[column], errors="coerce")
    valid = day_df[values.notna()].copy()
    if valid.empty:
        return set()
    keep = max(1, int(round(len(valid) * frac)))
    return set(valid.sort_values(column, ascending=False).head(keep)["day_idx"])


def summarize_subset(df, subset_name, day_ids):
    if day_ids is not None:
        df = df[df["day_idx"].isin(day_ids)]
    metrics = compute_metrics(df, include_mae=False)
    return {
        "subset": subset_name,
        "num_days": metrics["num_days"],
        "num_rows": metrics["num_rows"],
        "ic": metrics["ic"],
        "rankic": metrics["rankic"],
        "precision_at_10": metrics["precision_at_10"],
        "precision_at_20": metrics["precision_at_20"],
        "long_short_return": metrics["long_short_return"],
        "sharpe": metrics["sharpe"],
    }


def summarize_file(path, top_frac):
    df = pd.read_csv(path)
    if "split" in df.columns:
        df = df[df["split"] == "test"]
    if df.empty:
        return []
    for col in ["market_vol_lookback", "synchronism_lookback", "dispersion_lookback"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    dataset = str(df["dataset"].iloc[0])
    model = str(df["model"].iloc[0])
    regime_mode = frame_regime_mode(df)
    seed = int(df["seed"].iloc[0])
    stress_source = "lookback"
    if "stress_source" in df.columns and df["stress_source"].notna().any():
        stress_source = str(df["stress_source"].dropna().iloc[0])

    day_df = df[
        [
            "day_idx",
            "market_vol_lookback",
            "synchronism_lookback",
            "dispersion_lookback",
        ]
    ].drop_duplicates("day_idx")
    subsets = [
        ("all", None),
        ("high_vol", top_days(day_df, "market_vol_lookback", top_frac)),
        ("high_sync", top_days(day_df, "synchronism_lookback", top_frac)),
        ("high_dispersion", top_days(day_df, "dispersion_lookback", top_frac)),
    ]

    rows = []
    for subset_name, day_ids in subsets:
        row = {
            "dataset": dataset,
            "model": model,
            "regime_mode": regime_mode,
            "seed": seed,
            "stress_source": stress_source,
        }
        row.update(summarize_subset(df, subset_name, day_ids))
        rows.append(row)
    return rows


def main():
    args = parse_args()
    results_dir = get_results_dir(args)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(results_dir.glob("preds_*.csv")):
        rows.extend(summarize_file(path, args.top_frac))
    output = results_dir / "summary_stress.csv"
    pd.DataFrame(rows, columns=STRESS_COLUMNS).to_csv(output, index=False)
    print("Wrote {}".format(output))


if __name__ == "__main__":
    main()
