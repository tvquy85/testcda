import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RCLSProtoConfig:
    n_features: int = 5
    lookback: int = 16
    d_model: int = 64
    num_regimes: int = 2
    num_prototypes: int = 16
    dropout: float = 0.10
    alpha: float = 0.10
    prototype_delta_scale: float = 0.10
    temperature: float = 0.70
    lambda_uncertainty: float = 0.50
    use_uncertainty: bool = True
    static_proto: bool = False
    uniform_gate: bool = False
    architecture: str = "v1"
    decouple_rank_score: bool = False
    use_stress_aux: bool = False
    eps: float = 1e-6


class TemporalConvEncoder(nn.Module):
    def __init__(self, n_features: int, d_model: int, dropout: float):
        super().__init__()
        self.input = nn.Linear(n_features, d_model)
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.conv5 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        self.mix = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        hc = h.transpose(1, 2)
        h3 = self.conv3(hc).transpose(1, 2)
        h5 = self.conv5(hc).transpose(1, 2)
        pooled = torch.cat([h3.mean(dim=1), h5.mean(dim=1)], dim=-1)
        return self.norm(self.mix(pooled))


class RevINMultiScaleEncoder(nn.Module):
    def __init__(self, n_features: int, d_model: int, dropout: float, eps: float):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(n_features))
        self.affine_bias = nn.Parameter(torch.zeros(n_features))
        self.input = nn.Linear(n_features, d_model)
        specs = [(3, 1), (5, 2), (9, 4)]
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    d_model,
                    d_model,
                    kernel_size=kernel,
                    padding=dilation * (kernel - 1) // 2,
                    dilation=dilation,
                    groups=d_model,
                )
                for kernel, dilation in specs
            ]
        )
        self.mix = nn.Sequential(
            nn.Linear(len(specs) * 2 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
        x_norm = (x - mean) / std
        x_norm = x_norm * self.affine_weight.view(1, 1, -1) + self.affine_bias.view(1, 1, -1)
        h = self.input(x_norm)
        hc = h.transpose(1, 2)
        pooled = []
        for branch in self.branches:
            z = F.gelu(branch(hc)).transpose(1, 2)
            pooled.append(z.mean(dim=1))
            pooled.append(z.amax(dim=1))
        return self.norm(self.mix(torch.cat(pooled, dim=-1)))


class LookbackRegimeFeatures(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        price = torch.nan_to_num(x[:, :, -1].float(), nan=0.0, posinf=0.0, neginf=0.0)
        ret = (price[:, 1:] - price[:, :-1]) / (price[:, :-1].abs() + self.eps)
        ret = torch.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0).clamp(-0.5, 0.5)
        if mask is not None:
            valid = mask.view(-1) > 0.5
            if torch.any(valid):
                ret = ret[valid]

        market_ret = ret.mean(dim=0)
        market_ret_mean = market_ret.mean()
        market_ret_std = market_ret.std(unbiased=False)
        market_ret_last = market_ret[-1]
        downside = torch.minimum(market_ret, torch.zeros_like(market_ret))
        downside_vol = torch.sqrt((downside ** 2).mean() + self.eps)
        dispersion = ret.std(dim=0, unbiased=False).mean()
        synchronism = (torch.sign(ret) == torch.sign(market_ret).view(1, -1)).float().mean()
        mean_abs_ret = ret.abs().mean()
        max_abs_ret = ret.abs().amax()
        frac_positive = (ret > 0).float().mean()
        feats = torch.stack(
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
            ]
        )
        return torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


class RegimeRouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        regime_feat_dim: int,
        num_regimes: int,
        hidden: int,
        dropout: float,
    ):
        super().__init__()
        in_dim = 4 * d_model + regime_feat_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_regimes),
        )
        self.context = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def summarize_h(self, h: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                h.mean(dim=0),
                h.std(dim=0, unbiased=False),
                h.max(dim=0).values,
                h.min(dim=0).values,
            ],
            dim=-1,
        )

    def forward(
        self,
        h: torch.Tensor,
        regime_feats: torch.Tensor,
        temperature: float,
        uniform_gate: bool = False,
    ):
        z = torch.cat([self.summarize_h(h), regime_feats], dim=-1)
        logits = self.net(z)
        if uniform_gate:
            pi = torch.ones_like(logits) / float(logits.numel())
        else:
            pi = torch.softmax(logits / max(float(temperature), 1e-6), dim=-1)
        return pi, self.context(z), logits


class DynamicPrototypeBank(nn.Module):
    def __init__(
        self,
        num_regimes: int,
        num_prototypes: int,
        d_model: int,
        context_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.base = nn.Parameter(torch.randn(num_regimes, num_prototypes, d_model) * 0.02)
        self.delta = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim, num_regimes * num_prototypes * d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.num_regimes = num_regimes
        self.num_prototypes = num_prototypes
        self.d_model = d_model

    def forward(self, context: torch.Tensor, scale: float, static_proto: bool = False):
        if static_proto:
            p = self.base
            delta_norm = torch.zeros((), device=p.device, dtype=p.dtype)
        else:
            delta = self.delta(context).view(self.num_regimes, self.num_prototypes, self.d_model)
            p = self.base + scale * delta
            delta_norm = delta.norm()
        return self.norm(p), delta_norm


class PrototypeSetMixer(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.q1 = nn.Linear(d_model, d_model, bias=False)
        self.k1 = nn.Linear(d_model, d_model, bias=False)
        self.v1 = nn.Linear(d_model, d_model, bias=False)
        self.q2 = nn.Linear(d_model, d_model, bias=False)
        self.k2 = nn.Linear(d_model, d_model, bias=False)
        self.v2 = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.scale = math.sqrt(d_model)

    def forward(
        self,
        h: torch.Tensor,
        prototypes: torch.Tensor,
        pi: torch.Tensor,
        alpha: float,
    ):
        qh = self.q1(h)
        kp = self.k1(prototypes)
        logits_ep = torch.einsum("nd,kmd->knm", qh, kp) / self.scale
        attn_ep = torch.softmax(logits_ep, dim=-1)
        vh = self.v1(h)
        denom = attn_ep.sum(dim=1).clamp_min(1e-6)
        z = torch.einsum("knm,nd->kmd", attn_ep, vh) / denom.unsqueeze(-1)

        qh2 = self.q2(h)
        kz = self.k2(z)
        logits_pe = torch.einsum("nd,kmd->knm", qh2, kz) / self.scale
        attn_pe = torch.softmax(logits_pe, dim=-1)
        vz = self.v2(z)
        c = torch.einsum("knm,kmd->knd", attn_pe, vz)
        mixed = torch.einsum("k,knd->nd", pi, c)
        mixed = self.out(self.dropout(mixed))
        h_out = self.norm(h + alpha * mixed)
        return h_out, mixed.norm()


class ReliabilityRankingHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        use_uncertainty: bool,
        lambda_uncertainty: float,
        eps: float,
        decouple_rank_score: bool = False,
    ):
        super().__init__()
        self.mu = nn.Linear(d_model, 1)
        self.rank = nn.Linear(d_model, 1) if decouple_rank_score else None
        self.sigma = nn.Linear(d_model, 1)
        self.use_uncertainty = use_uncertainty
        self.lambda_uncertainty = lambda_uncertainty
        self.eps = eps
        self.decouple_rank_score = decouple_rank_score

    def forward(self, h: torch.Tensor):
        mu = self.mu(h).squeeze(-1)
        if self.use_uncertainty:
            sigma = F.softplus(self.sigma(h).squeeze(-1)) + self.eps
        else:
            sigma = torch.ones_like(mu)
        if self.decouple_rank_score:
            score = self.rank(h).squeeze(-1)
        elif self.use_uncertainty:
            score = mu / sigma.pow(self.lambda_uncertainty)
        else:
            score = mu
        return score, mu, sigma


class RCLSProtoRank(nn.Module):
    def __init__(self, cfg: RCLSProtoConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.architecture == "v2":
            self.encoder = RevINMultiScaleEncoder(cfg.n_features, cfg.d_model, cfg.dropout, cfg.eps)
        else:
            self.encoder = TemporalConvEncoder(cfg.n_features, cfg.d_model, cfg.dropout)
        self.regime_feats = LookbackRegimeFeatures(cfg.eps)
        self.router = RegimeRouter(
            cfg.d_model,
            regime_feat_dim=9,
            num_regimes=cfg.num_regimes,
            hidden=128,
            dropout=cfg.dropout,
        )
        self.proto_bank = DynamicPrototypeBank(
            cfg.num_regimes,
            cfg.num_prototypes,
            cfg.d_model,
            context_dim=128,
            dropout=cfg.dropout,
        )
        self.mixer = PrototypeSetMixer(cfg.d_model, cfg.dropout)
        self.head = ReliabilityRankingHead(
            cfg.d_model,
            cfg.use_uncertainty,
            cfg.lambda_uncertainty,
            cfg.eps,
            decouple_rank_score=cfg.decouple_rank_score,
        )
        self.aux_stress_head = nn.Linear(128, 2) if cfg.use_stress_aux else None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        manual_pi: Optional[torch.Tensor] = None,
        alpha_override: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        h = self.encoder(x)
        rf = self.regime_feats(x, mask)
        pi, context, logits = self.router(h, rf, self.cfg.temperature, self.cfg.uniform_gate)
        if manual_pi is not None:
            pi = manual_pi.to(device=h.device, dtype=h.dtype).view(-1)
            pi = pi / pi.sum().clamp_min(self.cfg.eps)
        prototypes, proto_delta_norm = self.proto_bank(
            context,
            self.cfg.prototype_delta_scale,
            self.cfg.static_proto,
        )
        alpha = self.cfg.alpha if alpha_override is None else float(alpha_override)
        h_out, delta_norm = self.mixer(h, prototypes, pi, alpha)
        score, mu, sigma = self.head(h_out)
        gate_entropy = -(pi * (pi + self.cfg.eps).log()).sum()
        proto_flat = F.normalize(prototypes.flatten(start_dim=0, end_dim=1), dim=-1)
        proto_gram = proto_flat @ proto_flat.t()
        proto_eye = torch.eye(proto_gram.shape[0], device=proto_gram.device, dtype=proto_gram.dtype)
        prototype_diversity = ((proto_gram * (1.0 - proto_eye)) ** 2).mean()
        aux_stress_logits = (
            self.aux_stress_head(context) if self.aux_stress_head is not None else None
        )
        return {
            "score": score,
            "rank_score": score,
            "mu": mu,
            "sigma": sigma,
            "pi": pi,
            "gate_logits": logits,
            "aux_stress_logits": aux_stress_logits,
            "gate_entropy": gate_entropy,
            "proto_delta_norm": proto_delta_norm,
            "delta_norm": delta_norm,
            "prototype_diversity": prototype_diversity,
            "regime_features": rf,
            "entity_pairwise_elements": torch.zeros((), device=x.device, dtype=x.dtype),
        }
