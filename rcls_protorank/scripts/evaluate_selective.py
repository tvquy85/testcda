import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from metrics import compute_metrics_from_frame  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate selective reliability for RCLS-ProtoRank.")
    parser.add_argument("--results-root", default=str(Path(__file__).resolve().parents[1] / "results"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--coverages", default="1.0,0.7,0.5,0.3")
    return parser.parse_args()


def load_predictions(results_root):
    frames = []
    for path in sorted(Path(results_root).glob("preds_*.csv")):
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def add_confidence(df):
    work = df.copy()
    if "sigma" in work.columns and work["sigma"].notna().any():
        sigma = pd.to_numeric(work["sigma"], errors="coerce")
        if sigma.std(skipna=True) > 1e-8:
            work["confidence_key"] = sigma
            work["confidence_mode"] = "sigma_low"
            return work
    if "gate_entropy" in work.columns and work["gate_entropy"].notna().any():
        entropy = pd.to_numeric(work["gate_entropy"], errors="coerce")
        if entropy.std(skipna=True) > 1e-8:
            work["confidence_key"] = entropy
            work["confidence_mode"] = "gate_entropy_low"
            return work
    if "regime_confidence" in work.columns and work["regime_confidence"].notna().any():
        conf = pd.to_numeric(work["regime_confidence"], errors="coerce")
        if conf.std(skipna=True) > 1e-8:
            work["confidence_key"] = -conf
            work["confidence_mode"] = "regime_confidence_high"
            return work
    work["confidence_key"] = -pd.to_numeric(work["pred"], errors="coerce").abs()
    work["confidence_mode"] = "abs_pred_high"
    return work


def select_by_coverage(df, coverage):
    if coverage >= 0.999:
        return df
    pieces = []
    for _, day in df.groupby("day_idx"):
        keep = max(1, int(np.ceil(len(day) * coverage)))
        pieces.append(day.sort_values("confidence_key", ascending=True).head(keep))
    return pd.concat(pieces, ignore_index=True) if pieces else df.iloc[0:0]


def main():
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    output = Path(args.output).resolve() if args.output else results_root / "summary_reliability.csv"
    coverages = [float(x.strip()) for x in args.coverages.split(",") if x.strip()]
    df = load_predictions(results_root)
    rows = []
    if not df.empty:
        df = add_confidence(df)
        mode = str(df["confidence_mode"].iloc[0])
        group_cols = ["dataset", "model", "seed", "split"]
        if "regime_mode" in df.columns:
            group_cols.insert(2, "regime_mode")
        if "run_tag" in df.columns:
            group_cols.insert(3 if "regime_mode" in df.columns else 2, "run_tag")
        for keys, group in df.groupby(group_cols):
            key_map = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
            if key_map["split"] != "test":
                continue
            for coverage in coverages:
                sub = select_by_coverage(group, coverage)
                metrics = compute_metrics_from_frame(sub)
                rows.append(
                    {
                        **key_map,
                        "coverage": coverage,
                        "confidence_mode": mode,
                        **metrics,
                    }
                )
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
