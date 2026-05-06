import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


MAIN_COLUMNS = [
    "dataset",
    "model",
    "regime_mode",
    "seed",
    "num_days",
    "num_rows",
    "ic",
    "rankic",
    "precision_at_10",
    "precision_at_20",
    "long_short_return",
    "sharpe",
    "mae",
]

GATE_COLUMNS = [
    "dataset",
    "model",
    "regime_mode",
    "seed",
    "split",
    "num_days",
    "num_regimes",
    "regime",
    "mean_prob",
    "std_prob",
    "min_prob",
    "max_prob",
    "dominant_count",
    "dominant_share",
    "entropy_mean",
    "entropy_std",
]

GATE_STRESS_COLUMNS = [
    "dataset",
    "model",
    "regime_mode",
    "seed",
    "split",
    "feature",
    "regime",
    "corr_pearson",
    "corr_spearman",
]

REFERENCE_COLUMNS = [
    "reference_name",
    "dataset",
    "model",
    "condition",
    "epochs",
    "runs",
    "required_gpu",
    "numpy_seed",
    "torch_seed",
    "main_mixer_activation",
    "scale_mixer_activation",
    "stock_activation",
    "ic",
    "ric_legacy",
    "precision_at_10",
    "sharpe5",
    "comparison_role",
]

COMPARISON_COLUMNS = [
    "dataset",
    "seed",
    "condition",
    "baseline_model",
    "baseline_regime_mode",
    "model",
    "regime_mode",
    "baseline_ic",
    "model_ic",
    "delta_ic",
    "baseline_rankic",
    "model_rankic",
    "delta_rankic",
    "baseline_precision_at_10",
    "model_precision_at_10",
    "delta_precision_at_10",
    "baseline_sharpe",
    "model_sharpe",
    "delta_sharpe",
    "note",
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


def frame_regime_mode(df):
    if "regime_mode" in df.columns and df["regime_mode"].notna().any():
        value = str(df["regime_mode"].dropna().iloc[0])
        return value if value else "legacy_delta"
    return "legacy_delta"


def result_stem(dataset, model, regime_mode, seed):
    if regime_mode == "legacy_delta":
        return "{}_{}_seed{}".format(model, dataset, seed)
    return "{}_{}_{}_seed{}".format(model, regime_mode, dataset, seed)


def safe_corr(a, b, method="pearson"):
    s1 = pd.Series(a)
    s2 = pd.Series(b)
    if s1.nunique(dropna=True) < 2 or s2.nunique(dropna=True) < 2:
        return np.nan
    return s1.corr(s2, method=method)


def entropy_from_probs(frame, regime_cols):
    values = frame[regime_cols].to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    row_sum = values.sum(axis=1, keepdims=True)
    values = np.divide(values, row_sum, out=np.zeros_like(values), where=row_sum > 0)
    values = np.clip(values, 1e-12, 1.0)
    return -(values * np.log(values)).sum(axis=1)


def precision_at_k(day_df, k):
    d = day_df.sort_values("pred", ascending=False).head(k)
    return float((d["target"] > 0).mean()) if len(d) else np.nan


def long_short_return(day_df, k=10):
    d = day_df.sort_values("pred", ascending=False)
    if len(d) < 2 * k:
        return np.nan
    return float(d.head(k)["target"].mean() - d.tail(k)["target"].mean())


def compute_metrics(df, include_mae=True):
    df = df.copy()
    df = df[pd.to_numeric(df["mask"], errors="coerce").fillna(0.0) > 0.5]
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna(subset=["pred", "target"])
    if df.empty:
        return {
            "num_days": 0,
            "num_rows": 0,
            "ic": np.nan,
            "rankic": np.nan,
            "precision_at_10": np.nan,
            "precision_at_20": np.nan,
            "long_short_return": np.nan,
            "sharpe": np.nan,
            "mae": np.nan,
        }

    ic = []
    rankic = []
    p10 = []
    p20 = []
    ls = []
    for _, day_df in df.groupby("day_idx"):
        ic.append(safe_corr(day_df["pred"], day_df["target"], "pearson"))
        rankic.append(safe_corr(day_df["pred"], day_df["target"], "spearman"))
        p10.append(precision_at_k(day_df, 10))
        p20.append(precision_at_k(day_df, 20))
        ls.append(long_short_return(day_df, 10))

    ls = np.asarray(ls, dtype=float)
    sharpe = np.nan
    if np.isfinite(ls).sum() > 1 and np.nanstd(ls) > 0:
        sharpe = np.nanmean(ls) / np.nanstd(ls) * np.sqrt(252.0)

    return {
        "num_days": int(df["day_idx"].nunique()),
        "num_rows": int(len(df)),
        "ic": np.nanmean(ic),
        "rankic": np.nanmean(rankic),
        "precision_at_10": np.nanmean(p10),
        "precision_at_20": np.nanmean(p20),
        "long_short_return": np.nanmean(ls),
        "sharpe": sharpe,
        "mae": float(np.mean(np.abs(df["pred"] - df["target"]))) if include_mae else np.nan,
    }


def summarize_prediction_file(path):
    df = pd.read_csv(path)
    if "split" in df.columns:
        df = df[df["split"] == "test"]
    if df.empty:
        return None
    dataset = str(df["dataset"].iloc[0])
    model = str(df["model"].iloc[0])
    regime_mode = frame_regime_mode(df)
    seed = int(df["seed"].iloc[0])
    metrics = compute_metrics(df)
    row = {"dataset": dataset, "model": model, "regime_mode": regime_mode, "seed": seed}
    row.update(metrics)
    return row


def write_gate_stats(path, results_dir):
    df = pd.read_csv(path)
    if "split" in df.columns:
        df = df[df["split"] == "test"]
    if df.empty:
        return []
    regime_cols = [c for c in ["regime_0", "regime_1", "regime_2"] if c in df.columns]
    if not regime_cols:
        return []
    for col in regime_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    active_cols = [col for col in regime_cols if df[col].notna().any()]
    if not active_cols:
        return []

    dataset = str(df["dataset"].iloc[0])
    model = str(df["model"].iloc[0])
    regime_mode = frame_regime_mode(df)
    seed = int(df["seed"].iloc[0])
    day_probs = df[["day_idx"] + active_cols].drop_duplicates("day_idx")
    ent = (
        pd.to_numeric(df[["day_idx", "gate_entropy"]].drop_duplicates("day_idx")["gate_entropy"], errors="coerce")
        if "gate_entropy" in df.columns
        else pd.Series(entropy_from_probs(day_probs, active_cols))
    )
    dominant = day_probs[active_cols].idxmax(axis=1)
    rows = []
    for col in active_cols:
        values = day_probs[col].dropna()
        count = int((dominant == col).sum())
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "regime_mode": regime_mode,
                "seed": seed,
                "split": "test",
                "num_days": int(day_probs["day_idx"].nunique()),
                "num_regimes": len(active_cols),
                "regime": col,
                "mean_prob": values.mean(),
                "std_prob": values.std(ddof=0),
                "min_prob": values.min(),
                "max_prob": values.max(),
                "dominant_count": count,
                "dominant_share": count / max(1, len(day_probs)),
                "entropy_mean": ent.mean(),
                "entropy_std": ent.std(ddof=0),
            }
        )
    output = results_dir / "gate_stats_{}.csv".format(
        result_stem(dataset, model, regime_mode, seed)
    )
    pd.DataFrame(rows, columns=GATE_COLUMNS).to_csv(output, index=False)
    return rows


def gate_stress_relation_rows(path):
    df = pd.read_csv(path)
    if "split" in df.columns:
        df = df[df["split"] == "test"].copy()
    if df.empty:
        return []
    regime_cols = [c for c in ["regime_0", "regime_1", "regime_2"] if c in df.columns]
    for col in regime_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    active_cols = [
        col
        for col in regime_cols
        if df[col].notna().any() and df[col].nunique(dropna=True) > 1
    ]
    if not active_cols:
        return []

    feature_cols = [
        "market_vol_lookback",
        "synchronism_lookback",
        "dispersion_lookback",
        "mean_abs_ret_lookback",
    ]
    available_features = [col for col in feature_cols if col in df.columns]
    for col in available_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    day_df = df[["day_idx"] + available_features + active_cols].drop_duplicates("day_idx")
    dataset = str(df["dataset"].iloc[0])
    model = str(df["model"].iloc[0])
    regime_mode = frame_regime_mode(df)
    seed = int(df["seed"].iloc[0])
    rows = []
    for feature in available_features:
        if day_df[feature].nunique(dropna=True) < 2:
            continue
        for regime_col in active_cols:
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "regime_mode": regime_mode,
                    "seed": seed,
                    "split": "test",
                    "feature": feature,
                    "regime": regime_col,
                    "corr_pearson": safe_corr(day_df[feature], day_df[regime_col], "pearson"),
                    "corr_spearman": safe_corr(day_df[feature], day_df[regime_col], "spearman"),
                }
            )
    return rows


def parse_baseline_reference(output_root):
    baseline_path = None
    for root in [output_root] + list(output_root.parents):
        candidate = root / "baseline_stockmixer_repro_seed1.md"
        if candidate.exists():
            baseline_path = candidate
            break
    if baseline_path is None:
        baseline_path = output_root.parent / "baseline_stockmixer_repro_seed1.md"
    if not baseline_path.exists():
        return []
    text = baseline_path.read_text(encoding="utf-8")

    def extract(pattern, default=""):
        match = re.search(pattern, text)
        return match.group(1).strip() if match else default

    return [
        {
            "reference_name": extract(r"baseline_name=([^\n]+)", "StockMixer-Repro Best Seed"),
            "dataset": extract(r"dataset=([^\n]+)", "NASDAQ"),
            "model": "stockmixer",
            "condition": "historical_controlled_100_epoch_reference",
            "epochs": extract(r"epochs=([^\n]+)", "100"),
            "runs": extract(r"runs=([^\n]+)", "1"),
            "required_gpu": extract(r"required_gpu=([^\n]+)", "RTX 3090"),
            "numpy_seed": extract(r"numpy_seed=([^\n]+)", "123456789"),
            "torch_seed": extract(r"torch_seed=([^\n]+)", "12345678"),
            "main_mixer_activation": extract(r"main_mixer_activation=([^\n]+)", "hardswish"),
            "scale_mixer_activation": extract(r"scale_mixer_activation=([^\n]+)", "gelu"),
            "stock_activation": extract(r"stock_activation=([^\n]+)", "hardswish"),
            "ic": extract(r"IC=([0-9.+-eE]+)", ""),
            "ric_legacy": extract(r"RIC=([0-9.+-eE]+)", ""),
            "precision_at_10": extract(r"Prec@10=([0-9.+-eE]+)", ""),
            "sharpe5": extract(r"SR=([0-9.+-eE]+)", ""),
            "comparison_role": (
                "historical_reference_only_not_used_for_60_epoch_pilot_gain"
            ),
        }
    ]


def write_reference_baseline(output_root, results_dir):
    rows = parse_baseline_reference(output_root)
    output = results_dir / "summary_reference_baseline.csv"
    pd.DataFrame(rows, columns=REFERENCE_COLUMNS).to_csv(output, index=False)
    print("Wrote {}".format(output))


def write_pilot_comparison(summary_rows, results_dir):
    if not summary_rows:
        output = results_dir / "summary_pilot_comparison.csv"
        pd.DataFrame(columns=COMPARISON_COLUMNS).to_csv(output, index=False)
        print("Wrote {}".format(output))
        return

    summary = pd.DataFrame(summary_rows)
    metadata_path = results_dir / "run_metadata.csv"
    metadata = pd.DataFrame()
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        if "regime_mode" not in metadata.columns:
            metadata["regime_mode"] = "legacy_delta"

    def run_condition(row):
        if metadata.empty:
            return "matched_results_dir"
        candidates = metadata[
            (metadata["dataset"].astype(str) == str(row["dataset"]))
            & (metadata["model"].astype(str) == str(row["model"]))
            & (metadata["regime_mode"].astype(str) == str(row.get("regime_mode", "legacy_delta")))
            & (metadata["seed"].astype(str) == str(row["seed"]))
        ]
        if candidates.empty:
            return "matched_results_dir"
        item = candidates.iloc[0]
        return "matched_pilot_{}epoch_patience{}".format(
            int(item.get("epochs", 0)),
            int(item.get("patience", 0)),
        )

    rows = []
    for (dataset, seed), group in summary.groupby(["dataset", "seed"]):
        baseline = group[group["model"] == "stockmixer"]
        if baseline.empty:
            continue
        baseline = baseline.iloc[0]
        comparator_rows = group[group["model"] != "stockmixer"]
        for _, model_row in comparator_rows.iterrows():
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "condition": run_condition(model_row),
                    "baseline_model": "stockmixer",
                    "baseline_regime_mode": baseline.get("regime_mode", "legacy_delta"),
                    "model": model_row["model"],
                    "regime_mode": model_row.get("regime_mode", "legacy_delta"),
                    "baseline_ic": baseline["ic"],
                    "model_ic": model_row["ic"],
                    "delta_ic": model_row["ic"] - baseline["ic"],
                    "baseline_rankic": baseline["rankic"],
                    "model_rankic": model_row["rankic"],
                    "delta_rankic": model_row["rankic"] - baseline["rankic"],
                    "baseline_precision_at_10": baseline["precision_at_10"],
                    "model_precision_at_10": model_row["precision_at_10"],
                    "delta_precision_at_10": (
                        model_row["precision_at_10"] - baseline["precision_at_10"]
                    ),
                    "baseline_sharpe": baseline["sharpe"],
                    "model_sharpe": model_row["sharpe"],
                    "delta_sharpe": model_row["sharpe"] - baseline["sharpe"],
                    "note": (
                        "Gain uses only same-condition StockMixer from this results dir; "
                        "historical 100-epoch baseline is reference-only."
                    ),
                }
            )

    output = results_dir / "summary_pilot_comparison.csv"
    pd.DataFrame(rows, columns=COMPARISON_COLUMNS).to_csv(output, index=False)
    print("Wrote {}".format(output))


def main():
    args = parse_args()
    results_dir = get_results_dir(args)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    gate_rows = []
    gate_stress_rows = []
    for path in sorted(results_dir.glob("preds_*.csv")):
        row = summarize_prediction_file(path)
        if row is not None:
            rows.append(row)
        gate_rows.extend(write_gate_stats(path, results_dir))
        gate_stress_rows.extend(gate_stress_relation_rows(path))
    output = results_dir / "summary_main.csv"
    pd.DataFrame(rows, columns=MAIN_COLUMNS).to_csv(output, index=False)
    print("Wrote {}".format(output))
    gate_output = results_dir / "gate_stats.csv"
    pd.DataFrame(gate_rows, columns=GATE_COLUMNS).to_csv(gate_output, index=False)
    print("Wrote {}".format(gate_output))
    gate_stress_output = results_dir / "gate_stress_relation.csv"
    pd.DataFrame(gate_stress_rows, columns=GATE_STRESS_COLUMNS).to_csv(
        gate_stress_output,
        index=False,
    )
    print("Wrote {}".format(gate_stress_output))
    write_pilot_comparison(rows, results_dir)
    write_reference_baseline(args.output_root, results_dir)


if __name__ == "__main__":
    main()
