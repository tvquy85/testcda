# RCLS-F Submission Pack

This directory is an isolated paper experiment pack built from the local StockMixer source.

## Layout

```text
code/src/      copied StockMixer code plus RCLS-F integration
scripts/       smoke, pilot, and summary scripts
results/       generated prediction and summary CSVs
logs/          generated training logs
document/      inspection notes, protocol, and final report template
```

## Quick Check

```powershell
powershell -ExecutionPolicy Bypass -File rcls_f_submission/scripts/run_smoke.ps1
```

## Pilot

```powershell
powershell -ExecutionPolicy Bypass -File rcls_f_submission/scripts/run_pilot_3090.ps1
```

The pilot runs `stockmixer`, `rcls_f_k1`, and `rcls_f_k3` on `SP500` and `NASDAQ` with seed 0.

## RCLS-Delta

RCLS-Delta is the stabilized follow-up to the direct RCLS-F replacement. It preserves the original `NoGraphMixer` path and adds a small regime-conditioned low-rank residual correction driven only by lookback features.

Config check:

```powershell
python rcls_f_submission/scripts/check_delta_config.py
```

Forward smoke:

```powershell
python rcls_f_submission/scripts/smoke_delta_forward.py
```

Smoke:

```powershell
powershell -ExecutionPolicy Bypass -File rcls_f_submission/scripts/run_rcls_delta_smoke.ps1
```

NASDAQ comparison:

```powershell
powershell -ExecutionPolicy Bypass -File rcls_f_submission/scripts/run_rcls_delta_nasdaq.ps1
```

Canonical NASDAQ Delta comparison:

```powershell
powershell -ExecutionPolicy Bypass -File rcls_f_submission/scripts/run_delta_nasdaq_3090.ps1
```

Warm-start/control run:

```powershell
powershell -ExecutionPolicy Bypass -File rcls_f_submission/scripts/run_delta_nasdaq_warmstart_3090.ps1
```

New diagnostics:

```text
results/gate_features_<model>_<dataset>_seed<seed>.csv
results/gate_stats.csv
results/gate_stress_relation.csv
results/summary_gate_behavior.csv
results/summary_gate_feature_corr.csv
```

## Baseline Alignment

The stored `baseline_stockmixer_repro_seed1.md` result is a NASDAQ StockMixer 100-epoch historical reference. Pilot gains are computed only against the pilot-generated `stockmixer` row under the same 60-epoch, patience-8 condition.

Relevant outputs:

```text
results/summary_main.csv
results/summary_pilot_comparison.csv
results/summary_reference_baseline.csv
```
