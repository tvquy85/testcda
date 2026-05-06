from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REGIME_COLS = ["regime_0", "regime_1", "regime_2"]
CONTEXT_COLS = [
    "ctx_mean_return",
    "ctx_momentum_slope",
    "ctx_realized_volatility",
    "ctx_dispersion",
    "ctx_pca_ratio",
    "ctx_synchronism",
    "ctx_downside_volatility",
    "ctx_mean_abs_return",
    "ctx_max_abs_return",
    "ctx_frac_positive",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    return parser.parse_args()


def corr(x, y, method="pearson"):
    x = pd.Series(x)
    y = pd.Series(y)
    if method == "spearman":
        x = x.rank(method="average")
        y = y.rank(method="average")
    if x.std() < 1e-12 or y.std() < 1e-12:
        return np.nan
    return float(x.corr(y))


def main():
    args = parse_args()
    stats_rows = []
    corr_rows = []
    subset_rows = []
    for path in sorted(args.results_root.glob("preds_*.csv")):
        df = pd.read_csv(path)
        df = df[df["split"] == "test"].copy()
        if df.empty:
            continue
        day = df.drop_duplicates("day_idx")
        base = {"dataset": day["dataset"].iloc[0], "model": day["model"].iloc[0], "seed": int(day["seed"].iloc[0])}
        for regime, col in enumerate(REGIME_COLS):
            if col not in day or day[col].notna().sum() == 0:
                continue
            values = day[col].dropna()
            dominant = day["dominant_regime"] == regime
            stats_rows.append(
                {
                    **base,
                    "regime": regime,
                    "mean_prob": values.mean(),
                    "std_prob": values.std(),
                    "min_prob": values.min(),
                    "max_prob": values.max(),
                    "dominant_share": float(dominant.mean()),
                }
            )
            for ctx in CONTEXT_COLS:
                if ctx in day:
                    corr_rows.append(
                        {
                            **base,
                            "context_feature": ctx,
                            "regime": regime,
                            "pearson_corr": corr(day[col], day[ctx], "pearson"),
                            "spearman_corr": corr(day[col], day[ctx], "spearman"),
                        }
                    )
        for subset, feature in {"all": None, "high_volatility": "ctx_realized_volatility", "high_dispersion": "ctx_dispersion"}.items():
            sub_day = day if feature is None else day[day[feature] >= day[feature].quantile(0.70)]
            for regime, col in enumerate(REGIME_COLS):
                if col not in sub_day or sub_day[col].notna().sum() == 0:
                    continue
                subset_rows.append(
                    {
                        **base,
                        "subset": subset,
                        "regime": regime,
                        "mean_prob": sub_day[col].mean(),
                        "dominant_share": float((sub_day["dominant_regime"] == regime).mean()),
                        "entropy_mean": sub_day["gate_entropy"].mean(),
                        "entropy_std": sub_day["gate_entropy"].std(),
                        "num_days": len(sub_day),
                    }
                )
    pd.DataFrame(stats_rows).to_csv(args.results_root / "gate_stats.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(args.results_root / "gate_context_corr.csv", index=False)
    pd.DataFrame(subset_rows).to_csv(args.results_root / "gate_by_stress_subset.csv", index=False)
    print(f"Wrote gate diagnostics under {args.results_root}")


if __name__ == "__main__":
    main()
