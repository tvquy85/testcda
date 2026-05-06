import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REGIME_COLUMNS = ["regime_0", "regime_1", "regime_2"]
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


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze RCLS-ProtoRank gate and prototype diagnostics.")
    parser.add_argument("--results-root", default=str(Path(__file__).resolve().parents[1] / "results"))
    return parser.parse_args()


def load_gate_files(results_root):
    frames = []
    for path in sorted(Path(results_root).glob("gate_proto_diagnostics_*.csv")):
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def corr(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    valid = a.notna() & b.notna()
    if valid.sum() < 3 or a[valid].nunique() < 2 or b[valid].nunique() < 2:
        return np.nan
    return float(a[valid].corr(b[valid], method="pearson"))


def main():
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    df = load_gate_files(results_root)
    gate_rows = []
    corr_rows = []
    proto_rows = []
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
            row = {
                **key_map,
                "num_days": int(group["day_idx"].nunique()),
                "gate_entropy_mean": float(group["gate_entropy"].mean()),
                "gate_entropy_std": float(group["gate_entropy"].std(ddof=0)),
            }
            for col in REGIME_COLUMNS:
                if col in group.columns:
                    row[f"{col}_mean"] = float(group[col].mean())
                    row[f"{col}_std"] = float(group[col].std(ddof=0))
            gate_rows.append(row)
            proto_rows.append(
                {
                    **key_map,
                    "proto_delta_norm_mean": float(group["proto_delta_norm"].mean()),
                    "proto_delta_norm_std": float(group["proto_delta_norm"].std(ddof=0)),
                    "delta_norm_mean": float(group["delta_norm"].mean()),
                    "delta_norm_std": float(group["delta_norm"].std(ddof=0)),
                }
            )
            for regime_col in REGIME_COLUMNS + ["gate_entropy"]:
                if regime_col not in group.columns:
                    continue
                for feature_col in FEATURE_COLUMNS:
                    if feature_col not in group.columns:
                        continue
                    corr_rows.append(
                        {
                            **key_map,
                            "gate_signal": regime_col,
                            "feature": feature_col,
                            "pearson_corr": corr(group[regime_col], group[feature_col]),
                        }
                    )
    pd.DataFrame(gate_rows).to_csv(results_root / "gate_stats.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(results_root / "gate_feature_corr.csv", index=False)
    pd.DataFrame(proto_rows).to_csv(results_root / "proto_stats.csv", index=False)
    print(f"Wrote diagnostics under {results_root}")


if __name__ == "__main__":
    main()
