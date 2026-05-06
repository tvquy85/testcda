# RCLS-F Pilot Report

## Repository / Setup

- Train entry: `rcls_f_submission/code/src/train.py`
- Dataset split source: inherited from StockMixer `MARKET_CONFIGS`
- Hardware:
- Python / PyTorch / CUDA:
- Git commit:

## Implemented Changes

- Original stock mixer location: `NoGraphMixer` in `StockMixer`.
- RCLS-F module: `rcls_f_submission/code/src/model_rcls.py`.
- Model variants:
  - `stockmixer`
  - `rcls_f_k1`
  - `rcls_f_k3`
- Prediction saving: `rcls_f_submission/results/preds_<model>_<dataset>_seed<seed>.csv`.

## Main Results

Paste or convert `results/summary_main.csv` here.

```text
Dataset | Model | IC | RankIC | P@10 | P@20 | Long-short | Sharpe | MAE
```

## Matched Pilot Comparison

Paste or convert `results/summary_pilot_comparison.csv` here.

Use this table for RCLS-F gains. The baseline in this table is the same-condition pilot `stockmixer` run, not the historical 100-epoch note.

```text
Dataset | Model | Delta IC | Delta RankIC | Delta P@10 | Delta Sharpe
```

## Historical NASDAQ Reference

Paste or convert `results/summary_reference_baseline.csv` here.

This is the `baseline_stockmixer_repro_seed1.md` 100-epoch StockMixer reference. It is calibration only and should not be mixed into 60-epoch pilot gain calculations.

```text
Reference | Dataset | Epochs | IC | Legacy RIC | P@10 | SR
```

## Stress Results

Paste or convert `results/summary_stress.csv` here.

```text
Dataset | Subset | StockMixer RankIC | RCLS-F K=3 RankIC | Gain | P@10 Gain
```

## Reliability / Gate Diagnostics

Paste or convert `results/summary_reliability.csv` and `results/gate_stats_*.csv` here.

```text
Dataset | Model | Confidence source | Coverage | RankIC | P@10 | MAE
```

```text
Dataset | Regime | Mean prob | Std prob | Dominant count | Dominant share
```

## Efficiency

Paste or convert `results/summary_efficiency.csv` here.

```text
Dataset | Model | Params | Train sec | Total sec | Max VRAM GB | Inference ms/day
```

## Interpretation

- Strongest evidence:
- Weakest evidence:
- Recommended paper angle:
- Go/no-go recommendation:
- Historical 100-epoch reference used only as calibration:

## Leakage / Reproducibility Statement

- Splits unchanged:
- Gate inputs:
- Stress labels:
- Transaction costs:
- Seeds:
- Pilot condition: 60 epochs, patience 8.
- Historical reference condition: NASDAQ StockMixer, 100 epochs, no RCLS-F.
- Logs:
