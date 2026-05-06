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

## Baseline Alignment

The stored `baseline_stockmixer_repro_seed1.md` result is a NASDAQ StockMixer 100-epoch historical reference. Pilot gains are computed only against the pilot-generated `stockmixer` row under the same 60-epoch, patience-8 condition.

Relevant outputs:

```text
results/summary_main.csv
results/summary_pilot_comparison.csv
results/summary_reference_baseline.csv
```
