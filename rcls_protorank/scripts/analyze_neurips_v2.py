import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "market_ret_mean",
    "market_ret_std",
    "market_ret_last",
    "downside_vol",
    "dispersion",
    "synchronism",
    "mean_abs_ret",
    "max_abs_ret",
    "frac_positive",
]
REGIME_COLUMNS = ["regime_0", "regime_1", "regime_2", "gate_entropy"]


def parse_args():
    parser = argparse.ArgumentParser(description="Build NeurIPS V2 aggregate summaries.")
    parser.add_argument(
        "--results-root",
        default=str(Path(__file__).resolve().parents[1] / "results" / "neurips_v2"),
    )
    return parser.parse_args()


def read_many(root, pattern):
    frames = []
    for path in sorted(Path(root).glob(pattern)):
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def corr(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    valid = a.notna() & b.notna()
    if valid.sum() < 3 or a[valid].nunique() < 2 or b[valid].nunique() < 2:
        return np.nan
    return float(a[valid].corr(b[valid], method="pearson"))


def write_v2_comparison(root):
    df = read_many(root, "run_summary_*.csv")
    if df.empty:
        out = pd.DataFrame()
    else:
        out = df.sort_values(["dataset", "test_ic", "test_p10"], ascending=[True, False, False])
    out.to_csv(root / "summary_v2_comparison.csv", index=False)
    out.to_csv(root / "summary_ablation.csv", index=False)
    keep = [
        "dataset",
        "model",
        "regime_mode",
        "run_tag",
        "seed",
        "num_params",
        "elapsed_seconds",
        "peak_vram_mb",
    ]
    out[[col for col in keep if col in out.columns]].to_csv(root / "summary_efficiency.csv", index=False)


def write_calibration(root):
    df = read_many(root, "preds_*.csv")
    rows = []
    if not df.empty:
        df = df[df["split"] == "test"].copy()
        for col in ["target", "mu", "sigma", "nll", "mask", "sigma_1x_cover", "sigma_2x_cover"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        group_cols = ["dataset", "model", "regime_mode", "run_tag", "seed"]
        group_cols = [col for col in group_cols if col in df.columns]
        for keys, group in df.groupby(group_cols):
            key_map = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
            valid = group[group["mask"] > 0.5] if "mask" in group.columns else group
            row = {
                **key_map,
                "num_rows": int(len(valid)),
                "mae_mu": float(np.nanmean(np.abs(valid["mu"] - valid["target"]))),
                "sigma_mean": float(np.nanmean(valid["sigma"])),
                "nll": float(np.nanmean(valid["nll"])) if "nll" in valid.columns else np.nan,
                "sigma_1x_coverage": float(np.nanmean(valid["sigma_1x_cover"]))
                if "sigma_1x_cover" in valid.columns
                else np.nan,
                "sigma_2x_coverage": float(np.nanmean(valid["sigma_2x_cover"]))
                if "sigma_2x_cover" in valid.columns
                else np.nan,
            }
            rows.append(row)
            if "regime_label" in valid.columns:
                for label, sub in valid.groupby("regime_label"):
                    rows.append(
                        {
                            **key_map,
                            "regime_label": int(label),
                            "num_rows": int(len(sub)),
                            "mae_mu": float(np.nanmean(np.abs(sub["mu"] - sub["target"]))),
                            "sigma_mean": float(np.nanmean(sub["sigma"])),
                            "nll": float(np.nanmean(sub["nll"])) if "nll" in sub.columns else np.nan,
                            "sigma_1x_coverage": float(np.nanmean(sub["sigma_1x_cover"]))
                            if "sigma_1x_cover" in sub.columns
                            else np.nan,
                            "sigma_2x_coverage": float(np.nanmean(sub["sigma_2x_cover"]))
                            if "sigma_2x_cover" in sub.columns
                            else np.nan,
                        }
                    )
    pd.DataFrame(rows).to_csv(root / "summary_calibration.csv", index=False)


def write_regime_semantics(root):
    df = read_many(root, "gate_proto_diagnostics_*.csv")
    rows = []
    if not df.empty:
        df = df[df["split"] == "test"].copy()
        group_cols = ["dataset", "model", "regime_mode", "run_tag", "seed"]
        group_cols = [col for col in group_cols if col in df.columns]
        for keys, group in df.groupby(group_cols):
            key_map = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
            labels = group["regime_label"].astype(int).values if "regime_label" in group.columns else []
            switches = int(np.sum(labels[1:] != labels[:-1])) if len(labels) > 1 else 0
            base = {
                **key_map,
                "num_days": int(group["day_idx"].nunique()),
                "gate_switch_count": switches,
                "gate_entropy_mean": float(pd.to_numeric(group["gate_entropy"], errors="coerce").mean()),
                "gate_entropy_std": float(pd.to_numeric(group["gate_entropy"], errors="coerce").std(ddof=0)),
            }
            for regime_col in REGIME_COLUMNS:
                if regime_col not in group.columns:
                    continue
                for feature_col in FEATURE_COLUMNS:
                    if feature_col in group.columns:
                        rows.append(
                            {
                                **base,
                                "gate_signal": regime_col,
                                "feature": feature_col,
                                "pearson_corr": corr(group[regime_col], group[feature_col]),
                            }
                        )
    pd.DataFrame(rows).to_csv(root / "summary_regime_semantics.csv", index=False)


def write_reference(root):
    rows = [
        {
            "source": "local_stockmixer_repro",
            "dataset": "NASDAQ",
            "ic": 0.03510,
            "p10": 0.49578,
            "p20": 0.50190,
            "long_short": 0.00342,
            "sharpe": 3.38,
            "mae": 0.01370,
        },
        {
            "source": "stockmixer_paper_nasdaq",
            "dataset": "NASDAQ",
            "ic": 0.043,
            "p10": 0.545,
            "note": "Paper Prec@N and legacy RIC are not directly comparable unless protocol is matched.",
        },
    ]
    pd.DataFrame(rows).to_csv(root / "summary_paper_reference.csv", index=False)


def main():
    args = parse_args()
    root = Path(args.results_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    write_v2_comparison(root)
    write_calibration(root)
    write_regime_semantics(root)
    write_reference(root)
    print(f"Wrote NeurIPS V2 summaries under {root}")


if __name__ == "__main__":
    main()
