# RCLS-F Experiment Protocol

## Goal

Validate whether replacing StockMixer's static stock-to-market-to-stock mixer with RCLS-F improves average performance, stress-regime robustness, or selective reliability on `NASDAQ` and `SP500`.

The controlled baseline note `baseline_stockmixer_repro_seed1.md` is a NASDAQ-only 100-epoch historical reference. It must not be mixed into 60-epoch pilot gain calculations.

## Smoke Run

PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File rcls_f_submission/scripts/run_smoke.ps1
```

Bash:

```bash
bash rcls_f_submission/scripts/run_smoke.sh
```

Acceptance:

- Training starts on `SP500`.
- `rcls_f_k3` runs for 2 epochs without shape errors.
- A prediction CSV is created.
- RCLS prediction rows include `regime_0`, `regime_1`, and `regime_2`.
- Summary CSVs are generated.

## Pilot Run

PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File rcls_f_submission/scripts/run_pilot_3090.ps1
```

Bash:

```bash
bash rcls_f_submission/scripts/run_pilot_3090.sh
```

Pilot matrix:

```text
datasets: NASDAQ, SP500
models: stockmixer, rcls_f_k1, rcls_f_k3
seed: 0
numpy_seed: 123456789
torch_seed: 12345678
epochs: 60
patience: 8
gpu requirement: RTX 3090
```

Activation configuration:

```text
activation=hardswish
main_mixer_activation=hardswish
scale_mixer_activation=gelu
stock_activation=hardswish
```

## Outputs

Prediction files:

```text
results/preds_stockmixer_SP500_seed0.csv
results/preds_rcls_f_k1_SP500_seed0.csv
results/preds_rcls_f_k3_SP500_seed0.csv
results/preds_stockmixer_NASDAQ_seed0.csv
results/preds_rcls_f_k1_NASDAQ_seed0.csv
results/preds_rcls_f_k3_NASDAQ_seed0.csv
```

Summary files:

```text
results/summary_main.csv
results/summary_pilot_comparison.csv
results/summary_reference_baseline.csv
results/summary_stress.csv
results/summary_reliability.csv
results/summary_efficiency.csv
results/run_metadata.csv
results/gate_stats_<dataset>_<model>_seed0.csv
```

Logs:

```text
logs/<model>_<dataset>_seed0.log
```

## Interpretation Rules

- Main same-condition comparison: use `summary_pilot_comparison.csv`, where gains are computed only against the pilot `stockmixer` row with the same dataset, seed, epochs, patience, activations, and split.
- Historical reference: use `summary_reference_baseline.csv` only as a calibration row for the NASDAQ 100-epoch StockMixer baseline.
- Strong success: `rcls_f_k3` improves RankIC or Precision@K over the same-condition pilot StockMixer and beats `rcls_f_k1`.
- Paper-useful robustness: overall metrics are similar, but high-volatility, high-synchronism, or high-dispersion subsets improve.
- Selective reliability angle: top-confidence subsets improve RankIC or Precision@K, or reduce MAE.
- Negative but useful result: `rcls_f_k1` beats `rcls_f_k3`, gates collapse, or runtime is too high; report this as evidence that dynamic gating needs stronger supervision.

Do not fabricate missing runs. If a pilot run fails, report the exact failure and keep completed CSVs intact.

## RCLS-Delta Follow-up

RCLS-Delta should be evaluated after the direct RCLS-F negative result. It keeps StockMixer's static population path and applies a small residual regime correction.

PowerShell smoke:

```powershell
powershell -ExecutionPolicy Bypass -File rcls_f_submission/scripts/run_rcls_delta_smoke.ps1
```

Bash smoke:

```bash
bash rcls_f_submission/scripts/run_rcls_delta_smoke.sh
```

NASDAQ RCLS-Delta matrix:

```text
models: stockmixer, rcls_delta_identity, rcls_delta_k1, rcls_delta_k2_uniform,
        rcls_delta_k2_nostress, rcls_delta_k2, rcls_delta_k3_uniform, rcls_delta_k3
seed: 0
epochs: 60
patience: 8
```

RCLS-Delta acceptance:

- `rcls_delta_identity` should match `stockmixer` up to small floating-point differences.
- `rcls_delta_k2` should not show a constant gate in `summary_gate_behavior.csv`.
- Gate-entropy selective evaluation is day-level; stock-row selection is used only for stock-level confidence such as `abs_pred`.
- Use `rcls_delta_k2_uniform` to test whether learned routing adds value over a uniform expert mixture.

## Optional Strict NASDAQ Appendix

To compare directly with the stored 100-epoch baseline note, run:

```powershell
powershell -ExecutionPolicy Bypass -File rcls_f_submission/scripts/run_nasdaq_strict_100.ps1
```

This runs `stockmixer`, `rcls_f_k1`, and `rcls_f_k3` on NASDAQ with `epochs=100`, `patience=0`, and the same baseline seeds/activations. Keep this appendix separate from the 60-epoch pilot results.
