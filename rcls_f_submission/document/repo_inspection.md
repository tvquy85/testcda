# RCLS-F Repository Inspection

## Repository / Setup

- Source repository: local StockMixer checkout.
- Submission pack root: `rcls_f_submission/`.
- Copied training entry: `rcls_f_submission/code/src/train.py`.
- Original root `src/` was not overwritten by this pack.
- Python observed during inspection: 3.11.5.
- PyTorch observed during inspection: 2.5.1+cu121.
- GPU observed during inspection: NVIDIA GeForce RTX 3090.

## Dataset Status

- `SP500` is available through `dataset/SP500/SP500.npy`, observed shape `[474, 2526, 5]`.
- `NASDAQ` is available through `dataset/NASDAQ/*.pkl`, observed shapes:
  - eod: `[1026, 1245, 5]`
  - mask: `[1026, 1245]`
  - gt: `[1026, 1245]`
  - price: `[1026, 1245]`
- `NYSE` is excluded from the pilot because `dataset/NYSE/*.pkl` files are currently 0 bytes.

## Model / Shape Findings

- Original stock mixing block: `NoGraphMixer` in `src/model.py`.
- Original model class: `StockMixer`.
- Local stock mixer input shape: `[N, D]`.
- With default settings:
  - `N = stock_num`
  - `D = time_steps * 2 + scale_dim = 16 * 2 + 8 = 40`
- StockMixer flow:
  - input: `[stock_num, lookback_length, feature_dim]`
  - temporal/multi-scale mixer output after `channel_fc`: `[stock_num, 40]`
  - stock mixer output: `[stock_num, 40]`
  - final `time_fc_`: `[stock_num, 1]`
- RCLS-F integration adapts `[N, D]` to the supported core math by treating it as a single panel sample.

## Implemented Variant Interface

- `--model stockmixer`: original `NoGraphMixer`.
- `--model rcls_f_k3`: regime-conditioned low-rank mixer with `K=3`.
- `--model rcls_f_k1`: low-rank ablation with `K=1`.
- RCLS gate probabilities are stored on `model.last_regime_prob` after forward.

## Evaluation / Prediction Saving

- Prediction CSVs are written under `rcls_f_submission/results/`.
- Prediction files include validation and test rows, with `split` distinguishing them.
- Main summaries use only `split == test`.
- Saved `pred` is the return ratio used by the evaluator, not raw predicted price.

## Leakage Statement

- RCLS-F gate is computed from post-encoder features derived from the current sample lookback only.
- Lookback stress labels are computed from historical `price_data` within the input window.
- Stress labels are post-hoc diagnostics and are never fed into training, validation, early stopping, or model input.
- Dataset splits follow the existing StockMixer split indices.
- Sharpe and long-short metrics are diagnostic/no-cost unless transaction costs are added later.
