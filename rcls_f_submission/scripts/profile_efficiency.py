import argparse
from pathlib import Path

import pandas as pd


EFFICIENCY_COLUMNS = [
    "dataset",
    "model",
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
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = args.output_root / "results"
    metadata_path = results_dir / "run_metadata.csv"
    output = results_dir / "summary_efficiency.csv"
    if not metadata_path.exists():
        pd.DataFrame(columns=EFFICIENCY_COLUMNS).to_csv(output, index=False)
        print("No run metadata found; wrote empty {}".format(output))
        return
    metadata = pd.read_csv(metadata_path)
    rows = metadata[EFFICIENCY_COLUMNS].copy()
    rows.to_csv(output, index=False)
    print("Wrote {}".format(output))


if __name__ == "__main__":
    main()
