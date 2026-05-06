import argparse
from pathlib import Path

import pandas as pd


EFFICIENCY_COLUMNS = [
    "dataset",
    "model",
    "regime_mode",
    "seed",
    "num_params",
    "train_time_sec",
    "total_time_sec",
    "max_vram_gb",
    "infer_time_ms_per_day",
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


def main():
    args = parse_args()
    results_dir = get_results_dir(args)
    results_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = results_dir / "run_metadata.csv"
    output = results_dir / "summary_efficiency.csv"
    if not metadata_path.exists():
        pd.DataFrame(columns=EFFICIENCY_COLUMNS).to_csv(output, index=False)
        print("No run metadata found; wrote empty {}".format(output))
        return
    metadata = pd.read_csv(metadata_path)
    if "regime_mode" not in metadata.columns:
        metadata["regime_mode"] = "legacy_delta"
    rows = metadata[EFFICIENCY_COLUMNS].copy()
    rows.to_csv(output, index=False)
    print("Wrote {}".format(output))


if __name__ == "__main__":
    main()
