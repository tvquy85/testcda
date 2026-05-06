# Cursor / Codex 5.5 Execution Spec
## RCLS-F: Regime-Conditioned Low-Rank Set Mixer on Local StockMixer Source

**Mission:** implement, run, and summarize a minimal but paper-useful RCLS-F experiment on top of the local StockMixer repository.

This spec is meant to be pasted directly into Cursor/Codex 5.5 as the implementation instruction. The local StockMixer source already exists. Your job is to inspect it, learn its actual architecture, patch it minimally, run controlled experiments, and produce reproducible results/logs/tables.

---

## 0. Non-negotiable objective

Validate this claim with the smallest credible experiment:

> StockMixer's static stock-to-market-to-stock mixer is strong on average but cannot adapt cross-sectional interactions across market regimes. RCLS-F keeps the original StockMixer temporal encoder and replaces only the static stock mixer with a regime-conditioned low-rank set mixer, then tests whether this improves robustness on high-volatility, high-synchronism, and high-dispersion market days.

Do **not** rewrite StockMixer. Do **not** build a new forecasting framework. Do **not** add Transformer/foundation baselines. Do **not** change the train/validation/test split. Do **not** use future target returns inside the model or gate.

The best deliverable is not a huge benchmark. The best deliverable is a clean, auditable result pack:

```text
results/*.csv
logs/*.log
scripts/*.sh
scripts/*.py
src/model_rcls.py or minimal patches to existing model files
```

---

## 1. Current research context

### 1.1 Domain painpoint

Financial panel forecasting is non-stationary. Cross-sectional relationships among stocks change between calm, high-volatility, market-wide selloff, sideways, sector-rotation, and high-dispersion regimes. A single static stock mixer tends to learn an average relation across regimes.

### 1.2 StockMixer baseline to preserve

StockMixer uses a lightweight MLP-style structure:

- indicator mixing;
- temporal/multi-scale mixing;
- stock-to-market-to-stock mixing.

RCLS-F should preserve everything except the final/static stock-mixing operator.

### 1.3 Static StockMixer stock mixer

The original operator is approximately:

```math
H_out = H + M_2 sigma(M_1 LN(H))
```

where:

- `H`: stock embeddings, usually `[N, D]` or `[B, N, D]`;
- `M1`: stock/entity to market prototype compression, `[m, N]`;
- `M2`: market prototype to stock/entity restoration, `[N, m]`;
- `m`: dataset-specific market dimension.

### 1.4 RCLS-F replacement

Replace the static mixer with:

```math
H_out = H + sum_k pi_k(z_t) M_2^(k) sigma(M_1^(k) LN(H))
```

where:

- `K`: number of latent regimes, default `K=3`;
- `pi(z_t)`: learned mixture weights from a permutation-invariant market summary;
- each regime has its own low-rank stock-to-market-to-stock mixer.

---

## 2. Deadline-driven execution rule

This is a 6-12 hour experiment sprint. Prioritize in this exact order:

1. Reproduce original StockMixer baseline.
2. Implement RCLS-F K=3.
3. Implement/enable RCLS-F K=1 ablation.
4. Save predictions and gate probabilities.
5. Produce main metrics table.
6. Produce stress-regime table.
7. Produce gate/reliability diagnostics.
8. Produce efficiency table.
9. Only then try uncertainty, NYSE, extra seeds, or extra ablations.

If a later step blocks progress, skip it and keep the result pack coherent.

---

## 3. Repository-first inspection protocol

Before editing code, inspect the local repository. Do not assume file names.

Run from repository root:

```bash
pwd
ls -la
find . -maxdepth 3 -type f | sort | sed 's#^./##' | head -250

# Identify model/training/data/eval files.
grep -R "StockMixer\|stockmixer\|Stock" -n . --exclude-dir=.git || true
grep -R "market" -n . --exclude-dir=.git || true
grep -R "class .*Model\|class .*Mixer\|class .*Stock" -n . --exclude-dir=.git || true
grep -R "def forward" -n src . --exclude-dir=.git || true
grep -R "def train\|optimizer\|loss" -n src . --exclude-dir=.git || true
grep -R "IC\|RankIC\|Precision\|Sharpe\|spearman" -n src . --exclude-dir=.git || true
```

Create a working branch/backup:

```bash
git status || true
git checkout -b rcls_f_experiment || true
mkdir -p experiments/rcls_f scripts results logs
cp -r src src_backup_before_rcls_f || true
```

Create an inspection note:

```text
experiments/rcls_f/repo_inspection.md
```

It must include:

```text
- entry training script
- dataset/config mechanism
- model class name(s)
- stock mixing block location
- input/output tensor shapes around stock mixing
- metric/evaluator location
- prediction saving location, if any
```

---

## 4. Deliverables checklist

At completion, produce at least:

```text
results/
  preds_stockmixer_<dataset>_seed0.csv
  preds_rcls_f_k3_<dataset>_seed0.csv
  preds_rcls_f_k1_<dataset>_seed0.csv
  summary_main.csv
  summary_stress.csv
  summary_reliability.csv
  summary_efficiency.csv
  gate_stats_<dataset>_rcls_f_k3_seed0.csv

logs/
  stockmixer_<dataset>_seed0.log
  rcls_f_k3_<dataset>_seed0.log
  rcls_f_k1_<dataset>_seed0.log

scripts/
  run_smoke.sh
  run_pilot_3090.sh
  run_full_3090.sh
  evaluate_stress.py
  evaluate_selective.py
  summarize_results.py
  profile_efficiency.py

src/
  model_rcls.py
  plus minimal integration patches
```

Minimum dataset target:

```text
S&P500 and NASDAQ
```

Optional after the minimum works:

```text
NYSE
extra seeds: 1, 2
uncertainty head
uniform gate ablation
```

---

## 5. Phase A — reproduce original StockMixer first

Do this before any RCLS-F code.

Find the original command. Examples:

```bash
python src/train.py --dataset S&P500
python train.py --dataset S&P500
python main.py --dataset S&P500
```

If the repo uses config files, inspect and use them. Do not invent a new config system unless needed.

Run S&P500 first. Save the log:

```bash
mkdir -p logs results
python <train_entry> <original_args_for_sp500> 2>&1 | tee logs/stockmixer_S&P500_seed0.log
```

Acceptance criteria:

```text
[ ] training starts
[ ] validation/evaluation runs
[ ] original metrics are printed or saved
[ ] no data split changes were made
```

If baseline fails:

1. fix environment/data path/import issue first;
2. record exact fix in `experiments/rcls_f/repo_inspection.md`;
3. do not implement RCLS-F until baseline can run or until the minimal blocker is clearly isolated.

---

## 6. Phase B — add model variant selection without breaking baseline

Add a minimal model selection path.

Preferred CLI:

```bash
python src/train.py --dataset S&P500 --model stockmixer --seed 0
python src/train.py --dataset S&P500 --model rcls_f_k3 --seed 0
python src/train.py --dataset S&P500 --model rcls_f_k1 --seed 0
```

Acceptable fallback if CLI is risky:

```python
# near top of train/config file
MODEL_NAME = "stockmixer"  # stockmixer, rcls_f_k3, rcls_f_k1
SEED = 0
```

Rules:

```text
[ ] model=stockmixer must keep original behavior
[ ] original training command must still work
[ ] no dataset split changes
[ ] no silent exception swallowing
```

---

## 7. Phase C — identify the original stock mixing block

Search for code patterns equivalent to:

```python
M1
M2
market
stock_to_market
market_to_stock
nn.Parameter(... market ... stock ...)
torch.matmul(...)
torch.einsum(...)
```

Typical targets:

```text
src/model.py
src/models.py
src/net.py
src/StockMixer.py
```

Instrument shapes temporarily with a debug flag:

```python
if getattr(self, "debug_shapes", False):
    print("[DEBUG] H before stock mix:", H.shape)
```

Run one tiny smoke step if possible. Record actual observed shape in:

```text
experiments/rcls_f/repo_inspection.md
```

Common shape cases:

```text
Case 1: H = [N, D]
Case 2: H = [B, N, D]
Case 3: H = [B, T, N, D]
Case 4: H = [B, D, N]
```

If Case 3/4 appears, adapt RCLS wrapper to flatten/transpose and restore shape without changing semantics.

---

## 8. Phase D — implement `RCLSStockMixing`

Create:

```text
src/model_rcls.py
```

Add this implementation, then adapt only if actual local shapes require it.

```python
import torch
import torch.nn as nn


class RCLSStockMixing(nn.Module):
    """
    Regime-Conditioned Low-Rank Set Mixer.

    Drop-in replacement for StockMixer's static stock-to-market-to-stock block.

    Input shape supported:
      - [N, D]
      - [B, N, D]

    If local StockMixer uses another layout, wrap before/after this module in the
    integration layer rather than changing the core math.
    """

    def __init__(
        self,
        n_stocks: int,
        d_model: int,
        market_dim: int,
        num_regimes: int = 3,
        dropout: float = 0.10,
        activation: str = "hardswish",
        gate_hidden: int = 128,
        return_gate: bool = True,
        uniform_gate: bool = False,
    ):
        super().__init__()
        self.n_stocks = int(n_stocks)
        self.d_model = int(d_model)
        self.market_dim = int(market_dim)
        self.num_regimes = int(num_regimes)
        self.return_gate = bool(return_gate)
        self.uniform_gate = bool(uniform_gate)

        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)

        act = activation.lower()
        if act == "hardswish":
            self.act = nn.Hardswish()
        elif act == "relu":
            self.act = nn.ReLU()
        elif act == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Regime-specific stock/entity -> market/population prototypes.
        # Shape: [K, m, N]
        self.M1 = nn.Parameter(torch.randn(self.num_regimes, self.market_dim, self.n_stocks) * 0.02)

        # Regime-specific market/population -> stock/entity restoration.
        # Shape: [K, N, m]
        self.M2 = nn.Parameter(torch.randn(self.num_regimes, self.n_stocks, self.market_dim) * 0.02)

        # Permutation-invariant panel summary: mean/std/max/min over stocks.
        self.gate = nn.Sequential(
            nn.Linear(4 * self.d_model, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, self.num_regimes),
        )

        self.last_regime_prob = None

    def _summary(self, Z: torch.Tensor) -> torch.Tensor:
        # Z: [B, N, D]
        mean = Z.mean(dim=1)
        std = Z.std(dim=1, unbiased=False)
        maxv = Z.max(dim=1).values
        minv = Z.min(dim=1).values
        return torch.cat([mean, std, maxv, minv], dim=-1)

    def forward(self, H: torch.Tensor):
        single = False
        if H.dim() == 2:
            H = H.unsqueeze(0)
            single = True
        if H.dim() != 3:
            raise ValueError(f"RCLSStockMixing expects [N,D] or [B,N,D], got {tuple(H.shape)}")

        B, N, D = H.shape
        if N != self.n_stocks:
            raise ValueError(f"Expected n_stocks={self.n_stocks}, got N={N}")
        if D != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got D={D}")

        Z = self.norm(H)

        if self.uniform_gate:
            pi = torch.full(
                (B, self.num_regimes),
                fill_value=1.0 / self.num_regimes,
                dtype=H.dtype,
                device=H.device,
            )
        else:
            stats = self._summary(Z)
            pi = torch.softmax(self.gate(stats), dim=-1)  # [B, K]

        # stock -> market: [B, K, m, D]
        U = torch.einsum("kmn,bnd->bkmd", self.M1, Z)
        U = self.act(U)

        # market -> stock: [B, K, N, D]
        O = torch.einsum("knm,bkmd->bknd", self.M2, U)

        # mixture over regimes: [B, N, D]
        O = (pi[:, :, None, None] * O).sum(dim=1)
        O = self.dropout(O)

        out = H + O
        self.last_regime_prob = pi.detach()

        if single:
            out = out.squeeze(0)
            pi = pi.squeeze(0)

        if self.return_gate:
            return out, pi
        return out
```

---

## 9. Phase E — integrate RCLS-F into StockMixer

Find where the original stock mixer is constructed. Keep original module for `model=stockmixer`.

Integration pattern:

```python
from model_rcls import RCLSStockMixing

if model_name in {"rcls_f", "rcls_f_k3", "rcls_f_k1"}:
    k = 3 if model_name in {"rcls_f", "rcls_f_k3"} else 1
    self.stock_mixing = RCLSStockMixing(
        n_stocks=n_stocks,
        d_model=d_model,
        market_dim=market_dim,
        num_regimes=k,
        dropout=dropout,
        gate_hidden=getattr(args, "gate_hidden", 128),
        return_gate=True,
    )
else:
    self.stock_mixing = OriginalStockMixing(...)
```

Forward pattern:

```python
mixed = self.stock_mixing(H)
if isinstance(mixed, tuple):
    H, pi = mixed
    self.last_regime_prob = pi.detach()
else:
    H = mixed
    self.last_regime_prob = None
```

If the local source does not have a standalone stock mixing class, create a subclass/wrapper:

```python
class RCLSStockMixer(OriginalStockMixer):
    def __init__(self, *args, num_regimes=3, **kwargs):
        super().__init__(*args, **kwargs)
        # replace only the stock-mixing members here
```

Do not change temporal encoder code unless the original code requires a call-site shape adapter.

---

## 10. Phase F — shape adapters if local source uses different layout

Use small adapters, not major rewrites.

### 10.1 If H is `[B, T, N, D]`

Apply RCLS on the final time or flatten time:

Preferred for speed and paper clarity:

```python
B, T, N, D = H.shape
H2 = H.reshape(B * T, N, D)
H2, pi = self.rcls_stock_mixing(H2)
H = H2.reshape(B, T, N, D)
pi = pi.reshape(B, T, -1)[:, -1, :]  # save final-step gate for logging
```

### 10.2 If H is `[B, D, N]`

```python
H_t = H.transpose(1, 2)       # [B, N, D]
H_t, pi = self.rcls_stock_mixing(H_t)
H = H_t.transpose(1, 2)       # [B, D, N]
```

### 10.3 If H is `[N, D]`

The provided module already supports it.

---

## 11. Phase G — save predictions and gate probabilities

Modify evaluator/test loop to save predictions. Minimum output path:

```text
results/preds_<model>_<dataset>_seed<seed>.csv
```

Minimum columns:

```csv
dataset,model,seed,day_idx,stock_idx,pred,target,mu,sigma,regime_0,regime_1,regime_2
```

Rules:

- If `mu`/`sigma` are unavailable, write empty values.
- If `K=1`, write only `regime_0` or write `regime_1`, `regime_2` as empty.
- If date/ticker identifiers are available, include them too:

```csv
date,ticker
```

Gate capture pattern:

```python
pi = getattr(model, "last_regime_prob", None)
```

If `pi` is per batch/day and predictions are per stock, repeat the same daily gate probabilities for all stocks from that day.

If evaluation batches multiple days, track batch/day indices carefully. If precise dates are not available, use monotonically increasing `day_idx`.

---

## 12. Phase H — metrics implementation

Create:

```text
scripts/summarize_results.py
```

It should read all `results/preds_*.csv` files and produce:

```text
results/summary_main.csv
```

Metrics:

```text
IC: Pearson correlation between pred and target per day, averaged over days
RankIC: Spearman correlation per day, averaged over days
Precision@10: fraction of positive targets among top 10 predicted stocks per day
Precision@20: fraction of positive targets among top 20 predicted stocks per day
Long-short return: mean(target of top K) - mean(target of bottom K)
Sharpe: mean(daily long-short return) / std(daily long-short return) * sqrt(252)
MAE: mean absolute error if meaningful
```

Implementation guidance:

```python
import numpy as np
import pandas as pd


def _safe_corr(a, b, method="pearson"):
    s1 = pd.Series(a)
    s2 = pd.Series(b)
    if s1.nunique(dropna=True) < 2 or s2.nunique(dropna=True) < 2:
        return np.nan
    return s1.corr(s2, method=method)


def precision_at_k(day_df, k):
    d = day_df.sort_values("pred", ascending=False).head(k)
    return float((d["target"] > 0).mean()) if len(d) else np.nan


def long_short_return(day_df, k=10):
    d = day_df.sort_values("pred", ascending=False)
    if len(d) < 2 * k:
        return np.nan
    return float(d.head(k)["target"].mean() - d.tail(k)["target"].mean())
```

Output columns:

```csv
dataset,model,seed,num_days,num_rows,ic,rankic,precision_at_10,precision_at_20,long_short_return,sharpe,mae
```

---

## 13. Phase I — stress-regime analysis

Create:

```text
scripts/evaluate_stress.py
```

Purpose: prove domain value under non-stationarity.

### 13.1 Preferred no-leakage stress labels

If the data loader exposes the input lookback returns/prices for each test day, compute stress labels from the lookback only:

```python
# past_returns shape example: [lookback, n_stocks]
market_ret = past_returns.mean(axis=1)
market_vol = market_ret.std()
dispersion = past_returns.std(axis=1).mean()
market_sign = np.sign(market_ret)[:, None]
stock_sign = np.sign(past_returns)
synchronism = (stock_sign == market_sign).mean()
```

High-stress subsets:

```text
high_vol: top 30% market_vol
high_sync: top 30% synchronism
high_dispersion: top 30% dispersion
```

### 13.2 Fallback ex-post stress labels

If lookback returns are not available quickly, compute stress labels from realized test-day target returns and label them explicitly as ex-post diagnostics.

For each `day_idx`:

```python
abs_market_target = abs(mean(target over stocks))
dispersion_target = std(target over stocks)
synchronism_target = fraction of stocks where sign(target_i) == sign(mean_target)
```

Subsets:

```text
ex_post_high_vol: top 30% abs_market_target
ex_post_high_sync: top 30% synchronism_target
ex_post_high_dispersion: top 30% dispersion_target
```

Hard rule:

```text
Ex-post stress labels may be used only for post-hoc analysis. Never feed them into the model, gate, train loop, validation, or early stopping.
```

### 13.3 Stress summary output

Produce:

```text
results/summary_stress.csv
```

Columns:

```csv
dataset,model,seed,stress_source,subset,num_days,num_rows,ic,rankic,precision_at_10,precision_at_20,long_short_return,sharpe
```

---

## 14. Phase J — gate diagnostics

For every RCLS-F prediction CSV, compute:

```text
gate entropy
dominant regime
mean/std/min/max probability per regime
regime usage counts
```

Create:

```text
results/gate_stats_<dataset>_<model>_seed<seed>.csv
```

Columns:

```csv
dataset,model,seed,regime,mean_prob,std_prob,min_prob,max_prob,dominant_count,dominant_share
```

Add this helper:

```python
import numpy as np


def entropy_from_probs(P):
    P = np.asarray(P, dtype=float)
    P = np.clip(P, 1e-12, 1.0)
    return -(P * np.log(P)).sum(axis=1)
```

If gate collapse occurs:

```text
symptom: dominant_share > 0.95 for one regime
```

Try one and only one quick mitigation:

```yaml
num_regimes: 2
gate_dropout: 0.0
entropy_regularizer: 0.001
```

Optional regularizer:

```python
def gate_regularizer(pi):
    # pi: [B, K]
    mean_pi = pi.mean(dim=0)
    global_entropy = -(mean_pi * (mean_pi + 1e-12).log()).sum()
    sample_entropy = -(pi * (pi + 1e-12).log()).sum(dim=1).mean()
    return -global_entropy + 0.1 * sample_entropy
```

Add to loss only if collapse is observed:

```python
loss = loss + 0.001 * gate_regularizer(pi)
```

---

## 15. Phase K — selective reliability analysis

Create:

```text
scripts/evaluate_selective.py
```

Output:

```text
results/summary_reliability.csv
```

Confidence sources:

1. If uncertainty head exists:

```python
confidence = -sigma
confidence_source = "sigma"
```

2. If no uncertainty head but gates exist:

```python
confidence = -gate_entropy
confidence_source = "gate_entropy"
```

3. Fallback:

```python
confidence = abs(pred)
confidence_source = "abs_pred"
```

Evaluate coverages:

```text
all / 100%
top 70%
top 50%
top 30%
```

Recommended implementation:

```python
# For each day, keep top coverage fraction by confidence, then compute day-level metrics.
```

Columns:

```csv
dataset,model,seed,confidence_source,coverage,num_days,num_rows,ic,rankic,precision_at_10,precision_at_20,mae
```

Interpretation rule:

```text
If top-confidence subsets improve RankIC/P@K or reduce MAE, this supports selective reliability even without a full calibrated uncertainty model.
```

---

## 16. Phase L — efficiency/profile logging

Create:

```text
scripts/profile_efficiency.py
```

Or integrate into training logs.

Required measurements:

```text
number of trainable parameters
training seconds per epoch
total training seconds
peak GPU memory GB
inference time per test day or per evaluation pass
```

Code snippets:

```python
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

```python
import time
import torch

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

start = time.time()
# run epoch or eval
elapsed = time.time() - start

if torch.cuda.is_available():
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
else:
    peak_gb = 0.0
```

Output:

```text
results/summary_efficiency.csv
```

Columns:

```csv
dataset,model,seed,num_params,train_time_sec,total_time_sec,max_vram_gb,infer_time_ms_per_day
```

---

## 17. Optional Phase M — uncertainty head

Implement only after RCLS-F K=3 and K=1 run cleanly.

Goal:

```text
mu: predicted return score
sigma: predictive uncertainty
ranking score: mu / sigma**lambda_uncertainty
```

Find the original final prediction layer. Add:

```python
self.mu_head = nn.Linear(final_dim, 1)
self.sigma_head = nn.Linear(final_dim, 1)
```

Forward:

```python
import torch.nn.functional as F

mu = self.mu_head(final_repr).squeeze(-1)
sigma = F.softplus(self.sigma_head(final_repr).squeeze(-1)) + 1e-6
score = mu / sigma.pow(lambda_uncertainty)
```

Loss:

```python
def laplace_nll(mu, sigma, y):
    return (torch.abs(y - mu) / sigma + torch.log(sigma)).mean()
```

Total loss:

```python
loss = original_loss(score, y) + beta_uncertainty * laplace_nll(mu, sigma, y)
```

Recommended:

```yaml
lambda_uncertainty: 0.5
beta_uncertainty: 0.03
```

Abort uncertainty immediately if:

```text
[ ] main RankIC drops badly
[ ] sigma explodes/collapses
[ ] training becomes unstable
```

The paper can still use gate entropy as reliability evidence.

---

## 18. Run scripts

### 18.1 `scripts/run_smoke.sh`

Create:

```bash
#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1

mkdir -p logs results

# Adapt train command to actual repo if needed.
python src/train.py \
  --dataset "S&P500" \
  --model "rcls_f_k3" \
  --seed 0 \
  --epochs 2 \
  --patience 1 \
  2>&1 | tee logs/smoke_rcls_f_k3_S&P500_seed0.log
```

### 18.2 `scripts/run_pilot_3090.sh`

Create:

```bash
#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1

mkdir -p logs results

DATASETS=("S&P500" "NASDAQ")
MODELS=("stockmixer" "rcls_f_k3" "rcls_f_k1")

for DATASET in "${DATASETS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    echo "Running ${MODEL} on ${DATASET}"
    python src/train.py \
      --dataset "${DATASET}" \
      --model "${MODEL}" \
      --seed 0 \
      --epochs 60 \
      --patience 8 \
      2>&1 | tee "logs/${MODEL}_${DATASET}_seed0.log"
  done
done

python scripts/summarize_results.py
python scripts/evaluate_stress.py
python scripts/evaluate_selective.py
python scripts/profile_efficiency.py || true
```

### 18.3 `scripts/run_full_3090.sh`

Create after pilot works:

```bash
#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1

mkdir -p logs results

DATASETS=("S&P500" "NASDAQ" "NYSE")
MODELS=("stockmixer" "rcls_f_k3" "rcls_f_k1")
SEEDS=(0)

for DATASET in "${DATASETS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      echo "Running ${MODEL} on ${DATASET}, seed=${SEED}"
      python src/train.py \
        --dataset "${DATASET}" \
        --model "${MODEL}" \
        --seed "${SEED}" \
        --epochs 80 \
        --patience 10 \
        2>&1 | tee "logs/${MODEL}_${DATASET}_seed${SEED}.log"
    done
  done
done

python scripts/summarize_results.py
python scripts/evaluate_stress.py
python scripts/evaluate_selective.py
python scripts/profile_efficiency.py || true
```

If the repo cannot support these CLI arguments, adapt the scripts to its actual config mechanism and document the adaptation.

---

## 19. Recommended configs

Dataset-specific market dimensions, if original StockMixer exposes them:

```yaml
NASDAQ:
  market_dim: 20
NYSE:
  market_dim: 25
S&P500:
  market_dim: 8
```

Default RCLS-F:

```yaml
num_regimes: 3
gate_hidden: 128
dropout: 0.10
activation: hardswish
uncertainty: false
```

Ablation:

```yaml
rcls_f_k1:
  num_regimes: 1
  uncertainty: false
```

Fast fallback if unstable:

```yaml
num_regimes: 2
gate_hidden: 64
dropout: 0.15
lr: 0.0005
uncertainty: false
```

---

## 20. Acceptance tests

### 20.1 Unit-ish test for RCLSStockMixing

Create a quick local check:

```python
from src.model_rcls import RCLSStockMixing
import torch

B, N, D, m, K = 4, 100, 64, 8, 3
mod = RCLSStockMixing(N, D, m, K)
H = torch.randn(B, N, D)
out, pi = mod(H)
assert out.shape == H.shape
assert pi.shape == (B, K)
assert torch.allclose(pi.sum(dim=1), torch.ones(B), atol=1e-5)
print("RCLSStockMixing shape test passed")
```

### 20.2 Smoke run

```bash
bash scripts/run_smoke.sh
```

Acceptance:

```text
[ ] no shape crash
[ ] loss decreases or at least remains finite
[ ] prediction CSV created
[ ] gate columns present for RCLS-F
```

### 20.3 Pilot run

```bash
bash scripts/run_pilot_3090.sh
```

Acceptance:

```text
[ ] StockMixer baseline exists for S&P500 and NASDAQ
[ ] RCLS-F K=3 exists for S&P500 and NASDAQ
[ ] RCLS-F K=1 exists for S&P500 and NASDAQ
[ ] summary_main.csv generated
[ ] summary_stress.csv generated
[ ] summary_reliability.csv generated
```

---

## 21. Go/no-go result interpretation

### Strong success

```text
RCLS-F K=3 > StockMixer on RankIC or P@K
RCLS-F K=3 > K=1
stress subsets improve more than all-day metrics
runtime < 2x StockMixer
```

Paper angle:

```text
Regime-conditioned low-rank set mixing improves average forecasting and stress-regime robustness.
```

### Good enough for a credible pilot

```text
overall metrics are similar
high-vol/high-sync/high-dispersion improves
or selective reliability improves using gate entropy
```

Paper angle:

```text
Average metrics hide regime-specific benefits; RCLS-F improves robustness/selective reliability under non-stationarity.
```

### Pivot case

```text
K=1 beats K=3 consistently
gate collapses
stress subsets do not improve
runtime is too high
```

Paper angle:

```text
Static low-rank mixing is a surprisingly strong regularizer; dynamic regime conditioning requires stronger or better-supervised gating.
```

Do not fabricate results. If a result is weak, report it honestly and use stress/reliability analysis to understand it.

---

## 22. Paper-ready tables to generate

### 22.1 Main table

```text
Dataset | Model | IC | RankIC | P@10 | P@20 | Long-short | Sharpe | Params | Runtime
```

Rows:

```text
S&P500 StockMixer
S&P500 RCLS-F K=1
S&P500 RCLS-F K=3
NASDAQ StockMixer
NASDAQ RCLS-F K=1
NASDAQ RCLS-F K=3
NYSE StockMixer optional
NYSE RCLS-F K=3 optional
```

### 22.2 Stress table

```text
Dataset | Subset | StockMixer RankIC | RCLS-F K=3 RankIC | Gain | P@10 Gain
```

Subsets:

```text
All
High-volatility
High-synchronism
High-dispersion
```

### 22.3 Gate diagnostics table

```text
Dataset | Regime | Mean prob | Std prob | Dominant count | Dominant share
```

### 22.4 Reliability table

```text
Dataset | Model | Confidence source | Coverage | RankIC | P@10 | MAE
```

Coverages:

```text
All
Top 70%
Top 50%
Top 30%
```

---

## 23. Reproducibility and NeurIPS-style reporting

Record in logs or `results/run_metadata.csv`:

```csv
dataset,model,seed,epochs,patience,learning_rate,batch_size,num_params,total_train_seconds,max_vram_gb,git_commit
```

Also record:

```text
- data split source
- lookback length
- target horizon
- whether stress labels are lookback-based or ex-post
- whether transaction cost is included in Sharpe/long-short returns
- hardware: RTX 3090
- Python/PyTorch/CUDA versions
```

Commands:

```bash
python --version
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
print('cuda', torch.version.cuda)
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
PY
git rev-parse HEAD || true
```

---

## 24. Leakage and finance validity rules

Hard rules:

```text
[ ] Gate input uses only model embeddings from the allowed lookback/input path.
[ ] Stress labels are not used for training/validation/early stopping.
[ ] Ex-post stress labels are clearly labeled ex_post_*.
[ ] No target returns are used as model input.
[ ] No test-period normalization using future statistics unless original StockMixer already did so.
[ ] If Sharpe uses no transaction costs, label it as diagnostic/no-cost.
```

Add a note to `experiments/rcls_f/repo_inspection.md`:

```text
Leakage statement:
- RCLS-F gate is computed from H after the original input encoder, using only current sample lookback features.
- Stress labels are post-hoc diagnostics and are not used by the model.
- Dataset splits are unchanged from StockMixer.
```

---

## 25. Fast troubleshooting guide

### Shape mismatch in einsum

Print:

```python
print('H', H.shape)
print('M1', self.M1.shape)
print('M2', self.M2.shape)
```

Expected:

```text
H:  [B, N, D]
M1: [K, m, N]
M2: [K, N, m]
U:  [B, K, m, D]
O:  [B, K, N, D]
```

### RCLS-F too slow

Try:

```yaml
num_regimes: 2
gate_hidden: 64
uncertainty: false
```

### Gate collapse

Try:

```yaml
num_regimes: 2
gate_dropout: 0.0
entropy_regularizer: 0.001
```

### Training unstable

Try:

```yaml
learning_rate: 0.0005
dropout: 0.15
gate_hidden: 64
uncertainty: false
```

### RCLS-F worse overall

Do not stop immediately. Check:

```text
summary_stress.csv
summary_reliability.csv
gate_stats_*.csv
```

The model may still provide stress-regime robustness or selective reliability.

---

## 26. Commit/checkpoint plan

Make small commits/checkpoints:

```bash
git add experiments/rcls_f/repo_inspection.md && git commit -m "inspect stockmixer repo for rcls" || true
git add src/model_rcls.py && git commit -m "add rcls stock mixing module" || true
git add src scripts && git commit -m "integrate rcls model variants and scripts" || true
git add results logs && git commit -m "add pilot results" || true
```

If git is unavailable, create tar snapshots:

```bash
tar -czf experiments/rcls_f/snapshot_after_rcls_impl.tgz src scripts
```

---

## 27. Final report file

Create:

```text
experiments/rcls_f/final_report.md
```

Use this structure:

```markdown
# RCLS-F Pilot Report

## Repository / setup
- train entry:
- dataset split source:
- hardware:
- torch/cuda:

## Implemented changes
- original stock mixer location:
- RCLS-F module:
- model variants:

## Main results
Paste summary_main.csv as markdown table.

## Stress results
Paste summary_stress.csv as markdown table.

## Reliability/gate diagnostics
Paste summary_reliability.csv and gate_stats as markdown tables.

## Efficiency
Paste summary_efficiency.csv.

## Interpretation
- strongest evidence:
- weakest evidence:
- go/no-go recommendation:

## Leakage/reproducibility statement
- splits unchanged:
- gate inputs:
- stress labels:
- seeds/logs:
```

---

## 28. Final success definition

A successful run produces:

```text
[ ] original StockMixer baseline reproduced
[ ] RCLS-F K=3 implemented and run
[ ] RCLS-F K=1 ablation implemented and run
[ ] predictions saved with target and gate probabilities
[ ] summary_main.csv
[ ] summary_stress.csv
[ ] summary_reliability.csv
[ ] summary_efficiency.csv or training-time/param logs
[ ] final_report.md
```

The final output should make it obvious whether the paper should emphasize:

```text
accuracy gain
stress-regime robustness
selective reliability
or a negative-but-informative regime-conditioning analysis
```

---

## 29. Last instruction to Cursor/Codex

Work source-first. Learn the actual StockMixer code before patching. Keep the original baseline intact. Make the smallest safe code changes. Save every prediction and log. Do not over-tune. Do not invent results. Optimize for a clean evidence pack that can be used immediately in a NeurIPS-style paper draft.
