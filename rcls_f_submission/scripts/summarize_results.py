import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


MAIN_COLUMNS = [
    "dataset",
    "model",
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
    "seed",
    "regime",
    "mean_prob",
    "std_prob",
    "min_prob",
    "max_prob",
    "dominant_count",
    "dominant_share",
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
    "model",
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
    return parser.parse_args()


def safe_corr(a, b, method="pearson"):
    s1 = pd.Series(a)
    s2 = pd.Series(b)
    if s1.nunique(dropna=True) < 2 or s2.nunique(dropna=True) < 2:
        return np.nan
    return s1.corr(s2, method=method)


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
    seed = int(df["seed"].iloc[0])
    metrics = compute_metrics(df)
    row = {"dataset": dataset, "model": model, "seed": seed}
    row.update(metrics)
    return row


def write_gate_stats(path, results_dir):
    df = pd.read_csv(path)
    if "split" in df.columns:
        df = df[df["split"] == "test"]
    regime_cols = [c for c in ["regime_0", "regime_1", "regime_2"] if c in df.columns]
    if not regime_cols:
        return
    for col in regime_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    active_cols = [col for col in regime_cols if df[col].notna().any()]
    if not active_cols:
        return

    dataset = str(df["dataset"].iloc[0])
    model = str(df["model"].iloc[0])
    seed = int(df["seed"].iloc[0])
    day_probs = df[["day_idx"] + active_cols].drop_duplicates("day_idx")
    dominant = day_probs[active_cols].idxmax(axis=1)
    rows = []
    for col in active_cols:
        values = day_probs[col].dropna()
        count = int((dominant == col).sum())
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "regime": col.replace("regime_", ""),
                "mean_prob": values.mean(),
                "std_prob": values.std(ddof=0),
                "min_prob": values.min(),
                "max_prob": values.max(),
                "dominant_count": count,
                "dominant_share": count / max(1, len(day_probs)),
            }
        )
    output = results_dir / "gate_stats_{}_{}_seed{}.csv".format(dataset, model, seed)
    pd.DataFrame(rows, columns=GATE_COLUMNS).to_csv(output, index=False)


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
    rows = []
    for (dataset, seed), group in summary.groupby(["dataset", "seed"]):
        baseline = group[group["model"] == "stockmixer"]
        if baseline.empty:
            continue
        baseline = baseline.iloc[0]
        for _, model_row in group[group["model"].isin(["rcls_f_k1", "rcls_f_k3"])].iterrows():
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "condition": "matched_pilot_60_epoch_patience8",
                    "baseline_model": "stockmixer",
                    "model": model_row["model"],
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
                        "Gain uses only same-condition pilot StockMixer; "
                        "historical 100-epoch baseline is reference-only."
                    ),
                }
            )

    output = results_dir / "summary_pilot_comparison.csv"
    pd.DataFrame(rows, columns=COMPARISON_COLUMNS).to_csv(output, index=False)
    print("Wrote {}".format(output))


def main():
    args = parse_args()
    results_dir = args.output_root / "results"
    rows = []
    for path in sorted(results_dir.glob("preds_*.csv")):
        row = summarize_prediction_file(path)
        if row is not None:
            rows.append(row)
        write_gate_stats(path, results_dir)
    output = results_dir / "summary_main.csv"
    pd.DataFrame(rows, columns=MAIN_COLUMNS).to_csv(output, index=False)
    print("Wrote {}".format(output))
    write_pilot_comparison(rows, results_dir)
    write_reference_baseline(args.output_root, results_dir)


if __name__ == "__main__":
    main()
