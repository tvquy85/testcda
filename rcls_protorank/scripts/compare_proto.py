import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Compare RCLS-ProtoRank runs from prediction CSV files.")
    parser.add_argument("--results-root", default=str(Path(__file__).resolve().parents[1] / "results"))
    parser.add_argument("--baseline-summary", default="")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_run_summaries(results_root):
    frames = []
    for path in sorted(Path(results_root).glob("run_summary_*.csv")):
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    output = Path(args.output).resolve() if args.output else results_root / "summary_proto_comparison.csv"
    df = load_run_summaries(results_root)
    rows = []
    if not df.empty:
        keep_cols = [
            "dataset",
            "model",
            "regime_mode",
            "run_tag",
            "seed",
            "best_epoch",
            "elapsed_seconds",
            "peak_vram_mb",
            "test_ic",
            "test_rankic",
            "test_p10",
            "test_p20",
            "test_long_short",
            "test_sharpe",
            "test_mae",
            "test_nll",
        ]
        rows.append(df[[col for col in keep_cols if col in df.columns]].copy())
    if args.baseline_summary:
        baseline = pd.read_csv(args.baseline_summary)
        baseline["source"] = "external_baseline"
        rows.append(baseline)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty and "source" not in out.columns:
        out["source"] = "rcls_protorank"
    elif not out.empty:
        out["source"] = out["source"].fillna("rcls_protorank")
    sort_cols = [col for col in ["dataset", "test_rankic", "test_ic"] if col in out.columns]
    if "test_rankic" in sort_cols:
        out = out.sort_values(["dataset", "test_rankic", "test_ic"], ascending=[True, False, False])
    elif sort_cols:
        out = out.sort_values(sort_cols)
    out.to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
