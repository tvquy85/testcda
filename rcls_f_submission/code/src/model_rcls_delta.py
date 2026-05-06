import torch
import torch.nn as nn
import torch.nn.functional as F


REGIME_FEATURE_NAMES = [
    "market_ret_mean",
    "market_ret_std",
    "market_ret_last",
    "downside_vol",
    "dispersion",
    "synchronism",
    "mean_abs_ret",
    "max_abs_ret",
    "frac_positive",
]


def build_activation(name):
    name = name.lower()
    if name == "hardswish":
        return nn.Hardswish()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError("Unsupported activation: {}".format(name))


def safe_lookback_regime_features(x, eps=1e-6):
    """Return lookback-only market state features from [N, T, F]."""
    price = x[:, :, -1].float()
    price = torch.nan_to_num(price, nan=0.0, posinf=0.0, neginf=0.0)

    prev = price[:, :-1]
    nxt = price[:, 1:]
    ret = (nxt - prev) / (prev.abs() + eps)
    ret = torch.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
    ret = ret.clamp(-0.5, 0.5)

    market_ret_t = ret.mean(dim=0)
    market_ret_mean = market_ret_t.mean()
    market_ret_std = market_ret_t.std(unbiased=False)
    market_ret_last = market_ret_t[-1]

    downside = torch.minimum(market_ret_t, torch.zeros_like(market_ret_t))
    downside_vol = torch.sqrt((downside ** 2).mean() + eps)

    dispersion_t = ret.std(dim=0, unbiased=False)
    dispersion = dispersion_t.mean()

    market_sign = torch.sign(market_ret_t).unsqueeze(0)
    stock_sign = torch.sign(ret)
    synchronism = (stock_sign == market_sign).float().mean()

    mean_abs_ret = ret.abs().mean()
    max_abs_ret = ret.abs().amax()
    frac_positive = (ret > 0).float().mean()

    return torch.stack(
        [
            market_ret_mean,
            market_ret_std,
            market_ret_last,
            downside_vol,
            dispersion,
            synchronism,
            mean_abs_ret,
            max_abs_ret,
            frac_positive,
        ],
        dim=0,
    )


def embedding_summary(h):
    mean = h.mean(dim=0)
    std = h.std(dim=0, unbiased=False)
    return torch.cat([mean, std], dim=-1)


class RCLSDeltaStockMixer(nn.Module):
    def __init__(
        self,
        base_stock_mixer,
        n_stocks,
        d_model,
        market_dim,
        num_regimes=2,
        gate_hidden=64,
        dropout=0.05,
        activation="hardswish",
        delta_scale=0.05,
        trainable_delta_scale=True,
        gate_temperature=0.7,
        gate_feature_mode="stress_embedding",
        uniform_gate=False,
    ):
        super(RCLSDeltaStockMixer, self).__init__()
        if gate_feature_mode not in ("stress_embedding", "embedding_only", "stress_only"):
            raise ValueError("Unsupported gate_feature_mode: {}".format(gate_feature_mode))
        if num_regimes <= 0:
            raise ValueError("num_regimes must be positive")

        self.base_stock_mixer = base_stock_mixer
        self.n_stocks = n_stocks
        self.d_model = d_model
        self.market_dim = market_dim
        self.num_regimes = num_regimes
        self.gate_temperature = gate_temperature
        self.gate_feature_mode = gate_feature_mode
        self.uniform_gate = uniform_gate

        self.stock_norm = nn.LayerNorm(n_stocks)
        self.activation = build_activation(activation)
        self.dropout = nn.Dropout(dropout)

        self.M1 = nn.Parameter(torch.randn(num_regimes, market_dim, n_stocks) * 0.005)
        self.M2 = nn.Parameter(torch.randn(num_regimes, n_stocks, market_dim) * 0.005)

        if trainable_delta_scale:
            self.delta_scale = nn.Parameter(torch.tensor(float(delta_scale)))
        else:
            self.register_buffer("delta_scale", torch.tensor(float(delta_scale)))

        gate_in_dim = 0
        if gate_feature_mode in ("stress_embedding", "stress_only"):
            gate_in_dim += len(REGIME_FEATURE_NAMES)
        if gate_feature_mode in ("stress_embedding", "embedding_only"):
            gate_in_dim += 2 * d_model

        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, num_regimes),
        )

        self.last_regime_prob = None
        self.last_gate_logits = None
        self.last_regime_features = None
        self.last_delta_norm = None
        self.last_base_norm = None

    def _gate_features(self, h, raw_x):
        features = []
        if self.gate_feature_mode in ("stress_embedding", "stress_only"):
            features.append(safe_lookback_regime_features(raw_x))
        if self.gate_feature_mode in ("stress_embedding", "embedding_only"):
            features.append(embedding_summary(h))
        return torch.cat(features, dim=0)

    def forward(self, h, raw_x, manual_pi=None):
        z_base = self.base_stock_mixer(h)

        gate_features = self._gate_features(h, raw_x)
        logits = self.gate(gate_features)
        if manual_pi is not None:
            pi = manual_pi.to(device=h.device, dtype=h.dtype).view(-1)
            pi = pi / pi.sum().clamp_min(1e-6)
        elif self.uniform_gate:
            pi = torch.ones_like(logits) / float(logits.numel())
        else:
            temperature = max(float(self.gate_temperature), 1e-6)
            pi = torch.softmax(logits / temperature, dim=-1)

        x = h.transpose(0, 1)
        x = self.stock_norm(x)
        u = torch.einsum("kmn,dn->kdm", self.M1, x)
        u = self.activation(u)
        o = torch.einsum("knm,kdm->kdn", self.M2, u)
        delta = (pi[:, None, None] * o).sum(dim=0).transpose(0, 1)
        delta = self.dropout(delta)
        z = z_base + self.delta_scale * delta

        denom = h.detach().norm() + 1e-6
        self.last_regime_prob = pi
        self.last_gate_logits = logits
        self.last_regime_features = gate_features.detach()
        self.last_delta_norm = delta.detach().norm() / denom
        self.last_base_norm = z_base.detach().norm() / denom
        return z

    def expert_diversity_loss(self):
        if self.num_regimes <= 1:
            return self.M1.sum() * 0.0
        flat = torch.cat(
            [self.M1.flatten(start_dim=1), self.M2.flatten(start_dim=1)],
            dim=1,
        )
        flat = F.normalize(flat, dim=1)
        gram = flat @ flat.t()
        eye = torch.eye(self.num_regimes, device=gram.device, dtype=gram.dtype)
        offdiag = gram * (1.0 - eye)
        return (offdiag ** 2).sum() / (self.num_regimes * (self.num_regimes - 1))
