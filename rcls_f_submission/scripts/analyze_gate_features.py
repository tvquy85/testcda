import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BEHAVIOR_COLUMNS = [
    "dataset",
    "model",
    "regime_mode",
    "seed",
    "num_days",
    "num_regimes",
    "entropy_mean",
    "entropy_std",
    "dominant_regime_0_share",
    "dominant_regime_1_share",
    "dominant_regime_2_share",
    "regime_0_mean",
    "regime_0_std",
    "regime_1_mean",
    "regime_1_std",
    "regime_2_mean",
    "regime_2_std",
]

CORR_COLUMNS = [
    "dataset",
    "model",
    "regime_mode",
    "seed",
    "feature",
    "regime",
    "pearson_corr",
    "spearman_corr",
]

FEATURE_ALIASES = {
    "market_vol_lookback": "market_ret_std",
    "synchronism_lookback": "synchronism",
    "dispersion_lookback": "dispersion",
    "mean_abs_ret_lookback": "mean_abs_ret",
}


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


def frame_regime_mode(df):
    if "regime_mode" in df.columns and df["regime_mode"].notna().any():
        value = str(df["regime_mode"].dropna().iloc[0])
        return value if value else "legacy_delta"
    return "legacy_delta"


def active_regime_cols(df):
    cols = [c for c in ["regime_0", "regime_1", "regime_2"] if c in df.columns]
    active = []
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].notna().any():
            active.append(col)
    return active


def entropy(frame, regime_cols):
    values = frame[regime_cols].to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    row_sum = values.sum(axis=1, keepdims=True)
    values = np.divide(values, row_sum, out=np.zeros_like(values), where=row_sum > 0)
    values = np.clip(values, 1e-12, 1.0)
    return -(values * np.log(values)).sum(axis=1)


def safe_corr(a, b, method):
    s1 = pd.to_numeric(a, errors="coerce")
    s2 = pd.to_numeric(b, errors="coerce")
    valid = s1.notna() & s2.notna()
    if valid.sum() < 3:
        return np.nan
    if s1[valid].nunique(dropna=True) < 2 or s2[valid].nunique(dropna=True) < 2:
        return np.nan
    return s1[valid].corr(s2[valid], method=method)


def summarize_behavior(df):
    if "split" in df.columns:
        df = df[df["split"] == "test"].copy()
    if df.empty:
        return None
    regime_cols = active_regime_cols(df)
    if not regime_cols:
        return None
    day_df = df.drop_duplicates("day_idx").copy()
    ent = (
        pd.to_numeric(day_df["gate_entropy"], errors="coerce")
        if "gate_entropy" in day_df.columns
        else pd.Series(entropy(day_df, regime_cols), index=day_df.index)
    )
    dominant = (
        pd.to_numeric(day_df["dominant_regime"], errors="coerce")
        if "dominant_regime" in day_df.columns
        else day_df[regime_cols].idxmax(axis=1).str.replace("regime_", "").astype(int)
    )
    row = {
        "dataset": str(day_df["dataset"].iloc[0]),
        "model": str(day_df["model"].iloc[0]),
        "regime_mode": frame_regime_mode(day_df),
        "seed": int(day_df["seed"].iloc[0]),
        "num_days": int(day_df["day_idx"].nunique()),
        "num_regimes": len(regime_cols),
        "entropy_mean": ent.mean(),
        "entropy_std": ent.std(ddof=0),
    }
    for idx in range(3):
        row["dominant_regime_{}_share".format(idx)] = float((dominant == idx).mean())
        col = "regime_{}".format(idx)
        if col in regime_cols:
            row["{}_mean".format(col)] = day_df[col].mean()
            row["{}_std".format(col)] = day_df[col].std(ddof=0)
        else:
            row["{}_mean".format(col)] = np.nan
            row["{}_std".format(col)] = np.nan
    return row


def summarize_corr(df):
    if "split" in df.columns:
        df = df[df["split"] == "test"].copy()
    if df.empty:
        return []
    regime_cols = active_regime_cols(df)
    if not regime_cols:
        return []
    day_df = df.drop_duplicates("day_idx").copy()
    dataset = str(day_df["dataset"].iloc[0])
    model = str(day_df["model"].iloc[0])
    regime_mode = frame_regime_mode(day_df)
    seed = int(day_df["seed"].iloc[0])
    rows = []
    for output_feature, source_col in FEATURE_ALIASES.items():
        if source_col not in day_df.columns:
            continue
        for regime_col in regime_cols:
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "regime_mode": regime_mode,
                    "seed": seed,
                    "feature": output_feature,
                    "regime": regime_col.replace("regime_", ""),
                    "pearson_corr": safe_corr(day_df[source_col], day_df[regime_col], "pearson"),
                    "spearman_corr": safe_corr(day_df[source_col], day_df[regime_col], "spearman"),
                }
            )
    return rows


def main():
    args = parse_args()
    results_dir = get_results_dir(args)
    results_dir.mkdir(parents=True, exist_ok=True)
    behavior_rows = []
    corr_rows = []
    for path in sorted(results_dir.glob("gate_features_*.csv")):
        df = pd.read_csv(path)
        behavior = summarize_behavior(df)
        if behavior is not None:
            behavior_rows.append(behavior)
        corr_rows.extend(summarize_corr(df))

    behavior_output = results_dir / "summary_gate_behavior.csv"
    corr_output = results_dir / "summary_gate_feature_corr.csv"
    pd.DataFrame(behavior_rows, columns=BEHAVIOR_COLUMNS).to_csv(
        behavior_output,
        index=False,
    )
    pd.DataFrame(corr_rows, columns=CORR_COLUMNS).to_csv(corr_output, index=False)
    print("Wrote {}".format(behavior_output))
    print("Wrote {}".format(corr_output))


if __name__ == "__main__":
    main()
