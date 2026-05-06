# RCLS-Delta Findings

## Purpose

RCLS-Delta is a stabilized residual version of RCLS for the fixed-universe StockMixer setting. It keeps the original StockMixer `NoGraphMixer` path and adds a small regime-conditioned low-rank correction.

## Required Tables

Populate from:

```text
results/summary_main.csv
results/summary_pilot_comparison.csv
results/summary_stress.csv
results/summary_reliability.csv
results/summary_gate_behavior.csv
results/summary_gate_feature_corr.csv
results/summary_efficiency.csv
```

```text
Dataset | Model | IC | RankIC | P@10 | Sharpe | High-vol RankIC | High-sync RankIC | Gate entropy std
```

## Identity Check

- StockMixer row:
- RCLS-Delta Identity row:
- Metric difference:
- Conclusion:

## Gate Behavior

- Entropy mean/std:
- Dominant regime shares:
- Regime probability std:
- Feature correlations:

## Interpretation

- If RCLS-Delta improves all-day and stress metrics, use it as the main method.
- If all-day metrics are similar but stress subsets improve, frame the result as robustness under non-stationary regimes.
- If gates are active but metrics do not improve, report it as a limitation and regularization target.
- If gates are constant, do not claim regime discovery.

## Reproducibility Notes

- Gate inputs are lookback-only.
- Pseudo-regime labels are fitted from train offsets only.
- Historical NASDAQ 100-epoch StockMixer remains a reference-only calibration row.
