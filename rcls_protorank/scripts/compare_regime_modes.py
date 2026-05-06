import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_WEIGHTS = {
    "test_ic": 0.35,
    "test_p10": 0.35,
    "test_long_short": 0.20,
    "test_p20": 0.10,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Rank RCLS-ProtoRank regime modes.")
    parser.add_argument("--results-root", default=str(Path(__file__).resolve().parents[1] / "results" / "regime_experiments"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def load_csvs(results_root, pattern):
    frames = []
    for path in sorted(Path(results_root).glob(pattern)):
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def zscore(series):
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return values * 0.0
    return (values - values.mean()) / std


def make_report(df, output):
    lines = [
        "# RCLS-ProtoRank Regime Mode Report",
        "",
        "Local RankIC is daily Spearman RankIC and is not directly comparable to the paper's legacy RIC scale.",
        "",
    ]
    if df.empty:
        lines.append("No regime mode runs were found.")
    else:
        cols = [
            "rank",
            "regime_mode",
            "model",
            "seed",
            "best_epoch",
            "test_ic",
            "test_p10",
            "test_p20",
            "test_long_short",
            "test_sharpe",
            "composite_score",
            "demoted",
            "demotion_reason",
        ]
        show = df[[col for col in cols if col in df.columns]].copy()
        headers = list(show.columns)
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in show.iterrows():
            values = []
            for col in headers:
                value = row[col]
                if isinstance(value, float):
                    values.append(f"{value:.6g}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        lines.extend(
            [
                "",
                "Reference points:",
                "",
                "- Current `rcls_proto_k2`: IC=0.031745, P@10=0.479747.",
                "- Current `rcls_proto_k2_no_uncert`: IC=0.020586, P@10=0.483544, P@20=0.496414.",
                "- StockMixer paper NASDAQ: IC=0.043, Prec@N=0.545.",
                "- Local StockMixer repro: IC=0.0366, Prec@10=0.527.",
            ]
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    output = Path(args.output).resolve() if args.output else results_root / "summary_regime_modes.csv"
    report = Path(args.report).resolve() if args.report else results_root / "regime_mode_report.md"

    summaries = load_csvs(results_root, "run_summary_*.csv")
    if summaries.empty:
        pd.DataFrame().to_csv(output, index=False)
        make_report(pd.DataFrame(), report)
        print(f"Wrote {output}")
        print(f"Wrote {report}")
        return

    gate_stats = results_root / "gate_stats.csv"
    if gate_stats.exists():
        gate = pd.read_csv(gate_stats)
        merge_cols = [col for col in ["dataset", "model", "regime_mode", "seed"] if col in summaries.columns and col in gate.columns]
        if merge_cols:
            summaries = summaries.merge(gate, on=merge_cols, how="left", suffixes=("", "_gate"))

    for col in METRIC_WEIGHTS:
        if col not in summaries.columns:
            summaries[col] = np.nan
    composite = pd.Series(np.zeros(len(summaries)), index=summaries.index, dtype=float)
    for col, weight in METRIC_WEIGHTS.items():
        composite = composite + weight * zscore(summaries[col])
    summaries["composite_score"] = composite

    reasons = []
    for _, row in summaries.iterrows():
        row_reasons = []
        min_occ = row.get("test_regime_occupancy_min", np.nan)
        max_occ = row.get("test_regime_occupancy_max", np.nan)
        switch_count = row.get("test_regime_switch_count", np.nan)
        num_days = row.get("test_num_days", row.get("num_days", np.nan))
        entropy_std = row.get("gate_entropy_std", np.nan)
        if np.isfinite(min_occ) and min_occ < 0.10:
            row_reasons.append("low_occupancy")
        if np.isfinite(max_occ) and max_occ > 0.95:
            row_reasons.append("dominant_single_regime")
        if np.isfinite(switch_count) and np.isfinite(num_days):
            if switch_count < 2 and row.get("regime_mode", "latent_current") != "latent_current":
                row_reasons.append("too_few_switches")
            if switch_count > 0.75 * num_days:
                row_reasons.append("too_many_switches")
        if np.isfinite(entropy_std) and entropy_std < 1e-6 and row.get("regime_mode", "latent_current") not in {"manual_breadth2", "manual_stress2"}:
            row_reasons.append("static_gate_entropy")
        reasons.append(",".join(row_reasons))
    summaries["demotion_reason"] = reasons
    summaries["demoted"] = summaries["demotion_reason"].astype(str).str.len() > 0
    summaries["rank_score"] = summaries["composite_score"] - summaries["demoted"].astype(float)
    summaries = summaries.sort_values(["rank_score", "test_ic", "test_p10"], ascending=[False, False, False]).reset_index(drop=True)
    summaries["rank"] = np.arange(1, len(summaries) + 1)
    summaries.to_csv(output, index=False)
    make_report(summaries, report)
    print(f"Wrote {output}")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
