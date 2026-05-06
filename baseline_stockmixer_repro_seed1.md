# StockMixer-Repro Best Seed

This file records the controlled single-seed baseline to use before comparing any improved model.

## Baseline Configuration

```text
baseline_name=StockMixer-Repro Best Seed
dataset=NASDAQ
required_gpu=RTX 3090
epochs=100
runs=1
numpy_seed=123456789
torch_seed=12345678
```

## Activation Configuration

```text
main_mixer_activation=hardswish
scale_mixer_activation=gelu
stock_activation=hardswish
```

## Baseline Command

```powershell
python src\train.py --market NASDAQ --epochs 100 --runs 1 --require-gpu 3090 `
  --activation hardswish `
  --main-mixer-activation hardswish `
  --scale-mixer-activation gelu `
  --stock-activation hardswish `
  --numpy-seed 123456789 `
  --torch-seed 12345678
```

## Baseline Result

```text
IC=0.0366
RIC=0.388
Prec@10=0.527
SR=1.540
```

## Comparison Notes

- Use this as the controlled single-seed baseline.
- Improved models must use the same seed, split, GPU, and training setup for the first comparison.
- If an improved model beats this single-seed baseline, run a 3-seed robustness check next.
- Report the single-seed comparison separately from any paper-style multi-run average.
