import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from metrics import compute_metrics_from_frame  # noqa: E402


STRESS_RULES = {
    "all": None,
    "high_vol": ("market_ret_std", 0.75),
    "high_sync": ("synchronism", 0.75),
    "high_dispersion": ("dispersion", 0.75),
    "high_abs_move": ("mean_abs_ret", 0.75),
    "downside": ("market_ret_last", 0.25),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RCLS-ProtoRank stress slices.")
    parser.add_argument("--results-root", default=str(Path(__file__).resolve().parents[1] / "results"))
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_predictions(results_root):
    frames = []
    for path in sorted(Path(results_root).glob("preds_*.csv")):
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def stress_subset(df, name, rule):
    if rule is None:
        return df
    column, q = rule
    if column not in df.columns:
        return df.iloc[0:0]
    values = df[["day_idx", column]].drop_duplicates()
    if values.empty:
        return df.iloc[0:0]
    threshold = values[column].quantile(q)
    if name == "downside":
        days = values[values[column] <= threshold]["day_idx"]
    else:
        days = values[values[column] >= threshold]["day_idx"]
    return df[df["day_idx"].isin(days)]


def main():
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    output = Path(args.output).resolve() if args.output else results_root / "summary_stress.csv"
    df = load_predictions(results_root)
    rows = []
    if not df.empty:
        group_cols = ["dataset", "model", "seed", "split"]
        if "regime_mode" in df.columns:
            group_cols.insert(2, "regime_mode")
        if "run_tag" in df.columns:
            group_cols.insert(3 if "regime_mode" in df.columns else 2, "run_tag")
        for keys, group in df.groupby(group_cols):
            key_map = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
            if key_map["split"] != "test":
                continue
            for stress_name, rule in STRESS_RULES.items():
                sub = stress_subset(group, stress_name, rule)
                metrics = compute_metrics_from_frame(sub)
                rows.append(
                    {
                        **key_map,
                        "stress_slice": stress_name,
                        **metrics,
                    }
                )
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
