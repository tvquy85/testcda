# RCLS NeurIPS Paper15 Seed1-Match Brief

## Setup
- Dataset: NASDAQ
- Epochs: 15
- RNG: numpy_seed=123456789, torch_seed=12345678, run label seed=0
- Activations: activation=hardswish, main=hardswish, scale=gelu, stock=hardswish
- Context: lookback-only, train-fitted minmax normalizer, context_mode=five for model input; all 10 context features are logged.

## Main Test Metrics
| model                 |       ic |   rankic |     icir |      p10 |      p20 |   long_short10 |      mae |
|:----------------------|---------:|---------:|---------:|---------:|---------:|---------------:|---------:|
| stockmixer            | 0.024077 | 0.034123 | 0.290145 | 0.496203 | 0.485232 |       0.001470 | 0.022942 |
| cag_mlp               | 0.027044 | 0.033970 | 0.336684 | 0.502954 | 0.492616 |       0.002015 | 0.019256 |
| rcls_cag_k1           | 0.027044 | 0.033970 | 0.336684 | 0.502954 | 0.492616 |       0.002015 | 0.019256 |
| rcls_cag_k2           | 0.022733 | 0.033694 | 0.288487 | 0.478059 | 0.487764 |       0.000400 | 0.027479 |
| rcls_cag_k2_nocontext | 0.023240 | 0.030304 | 0.297256 | 0.490717 | 0.491983 |       0.001084 | 0.028125 |
| rcls_cag_k2_nogate    | 0.016237 | 0.027240 | 0.194707 | 0.479747 | 0.485021 |       0.000028 | 0.030555 |
| rcls_cag_k2_uniform   | 0.005164 | 0.012016 | 0.074493 | 0.489030 | 0.481646 |       0.000708 | 0.055113 |

## Gate Notes
| model                 |   regime |   mean_prob |   std_prob |   dominant_share |
|:----------------------|---------:|------------:|-----------:|-----------------:|
| cag_mlp               |        0 |    1.000000 |   0.000000 |         1.000000 |
| rcls_cag_k1           |        0 |    1.000000 |   0.000000 |         1.000000 |
| rcls_cag_k2           |        0 |    0.985370 |   0.027916 |         1.000000 |
| rcls_cag_k2           |        1 |    0.014630 |   0.027916 |         0.000000 |
| rcls_cag_k2_nocontext |        0 |    0.583205 |   0.136188 |         0.734177 |
| rcls_cag_k2_nocontext |        1 |    0.416795 |   0.136188 |         0.265823 |
| rcls_cag_k2_nogate    |        0 |    0.985353 |   0.027016 |         1.000000 |
| rcls_cag_k2_nogate    |        1 |    0.014647 |   0.027016 |         0.000000 |
| rcls_cag_k2_uniform   |        0 |    0.500000 |   0.000000 |         1.000000 |
| rcls_cag_k2_uniform   |        1 |    0.500000 |   0.000000 |         0.000000 |
| stockmixer            |        0 |    1.000000 |   0.000000 |         1.000000 |

## Efficiency
| model                 |   num_params |   train_time_sec |   max_vram_gb |   model_size_mb |
|:----------------------|-------------:|-----------------:|--------------:|----------------:|
| stockmixer            |        45077 |       296.193209 |      0.042990 |        0.171955 |
| cag_mlp               |       195813 |       316.417485 |      0.045358 |        0.746967 |
| rcls_cag_k1           |       195813 |       313.975639 |      0.045358 |        0.746967 |
| rcls_cag_k2           |       139557 |       338.266987 |      0.045004 |        0.532368 |
| rcls_cag_k2_uniform   |       139557 |       317.971782 |      0.044611 |        0.532368 |
| rcls_cag_k2_nocontext |       139557 |       327.004891 |      0.044993 |        0.532368 |
| rcls_cag_k2_nogate    |       139557 |       321.291188 |      0.044980 |        0.532368 |

## Reference Rows
| reference             | dataset   |       ic |     icir |      p10 |   sharpe_legacy |
|:----------------------|:----------|---------:|---------:|---------:|----------------:|
| stockmixer_seed1_100e | NASDAQ    | 0.036600 | 0.388000 | 0.527000 |        1.540000 |
| cagclean_seed1_100e   | NASDAQ    | 0.032550 | 0.418030 | 0.518570 |        0.686930 |

## Interpretation
- `cag_mlp` and `rcls_cag_k1` are identical in this compatibility pilot, satisfying the K1 reproduction gate.
- `cag_mlp`/`rcls_cag_k1` beat the 15e StockMixer run on IC, ICIR and P@10, but StockMixer has slightly higher true RankIC in this short run.
- `rcls_cag_k2` does not beat K1 in average metrics at 15 epochs; its learned gate collapses mostly to regime 0, so the gate-alive criterion is weak.
- `rcls_cag_k2_nocontext` has more balanced regime usage, but weaker metrics than K1. This suggests the current context-conditioned router objective needs tuning before full runs.
- Use these outputs for paper structure and diagnostics only; do not claim full performance until 80/100e confirmation.