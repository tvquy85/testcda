# Baseline Alignment Note

## Controlled Baseline Reference

`baseline_stockmixer_repro_seed1.md` records:

```text
dataset=NASDAQ
required_gpu=RTX 3090
epochs=100
runs=1
numpy_seed=123456789
torch_seed=12345678
main_mixer_activation=hardswish
scale_mixer_activation=gelu
stock_activation=hardswish
IC=0.0366
RIC=0.388
Prec@10=0.527
SR=1.540
```

This row is a historical controlled StockMixer reference. It is not the direct comparator for the 60-epoch RCLS-F pilot.

## Matched Pilot Comparator

The same-condition RCLS-F pilot comparator is the pilot-generated `stockmixer` row with:

```text
datasets=NASDAQ,SP500
models=stockmixer,rcls_f_k1,rcls_f_k3
seed=0
numpy_seed=123456789
torch_seed=12345678
epochs=60
patience=8
activation=hardswish
main_mixer_activation=hardswish
scale_mixer_activation=gelu
stock_activation=hardswish
learning_rate=0.001
alpha=0.1
lookback_length=16
steps=1
fea_num=5
gpu_requirement=RTX 3090
```

Use `results/summary_pilot_comparison.csv` for gains. That file compares `rcls_f_k1` and `rcls_f_k3` only against the same-condition pilot `stockmixer`.

## Reporting Language

Use:

```text
Under matched pilot settings, RCLS-F is compared against a StockMixer rerun with identical seed, split, activations, epoch budget, and early-stopping rule.
```

Use:

```text
The previously recorded 100-epoch NASDAQ StockMixer result is included as a historical controlled reference, not as the denominator for pilot gains.
```
