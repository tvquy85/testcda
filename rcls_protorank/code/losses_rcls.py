from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class RCLSLossConfig:
    loss_huber_weight: float = 1.0
    loss_listwise_weight: float = 0.20
    loss_nll_weight: float = 0.05
    loss_ic_weight: float = 0.0
    loss_topk_weight: float = 0.0
    loss_calibration_weight: float = 0.0
    loss_proto_diversity_weight: float = 0.0
    loss_gate_balance_weight: float = 0.001
    loss_gate_confidence_weight: float = 0.0005
    rank_loss: str = "listnet"
    listnet_temperature: float = 1.0
    topk_loss_k: int = 10
    huber_delta: float = 1.0
    use_uncertainty: bool = True
    eps: float = 1e-6


def flatten_valid(x, mask):
    x = x.view(-1)
    if mask is None:
        return x
    m = mask.view(-1) > 0.5
    return x[m]


def masked_huber_loss(mu, target, mask=None, delta=1.0):
    mu_v = flatten_valid(mu, mask)
    y_v = flatten_valid(target, mask)
    if mu_v.numel() == 0:
        return mu.sum() * 0.0
    return F.huber_loss(mu_v, y_v, delta=delta)


def laplace_nll(mu, sigma, target, mask=None, eps=1e-6):
    mu_v = flatten_valid(mu, mask)
    sig_v = flatten_valid(sigma, mask).clamp_min(eps)
    y_v = flatten_valid(target, mask)
    if mu_v.numel() == 0:
        return mu.sum() * 0.0
    return (torch.abs(y_v - mu_v) / sig_v + torch.log(sig_v)).mean()


def listnet_loss(score, target, mask=None, temperature=1.0):
    s = flatten_valid(score, mask) / temperature
    y = flatten_valid(target, mask) / temperature
    if s.numel() <= 1:
        return score.sum() * 0.0
    p_y = torch.softmax(y, dim=0)
    log_p_s = torch.log_softmax(s, dim=0)
    return -(p_y.detach() * log_p_s).sum()


def listmle_loss(score, target, mask=None):
    s = flatten_valid(score, mask)
    y = flatten_valid(target, mask)
    if s.numel() <= 1:
        return score.sum() * 0.0
    order = torch.argsort(y, descending=True)
    s_sorted = s[order]
    log_cumsum = torch.logcumsumexp(s_sorted.flip(0), dim=0).flip(0)
    return (log_cumsum - s_sorted).mean()


def negative_pearson_loss(score, target, mask=None, eps=1e-6):
    s = flatten_valid(score, mask)
    y = flatten_valid(target, mask)
    if s.numel() <= 2:
        return score.sum() * 0.0
    s = s - s.mean()
    y = y - y.mean()
    denom = s.norm() * y.norm()
    if float(denom.detach().cpu()) <= eps:
        return score.sum() * 0.0
    return -(s * y).sum() / denom.clamp_min(eps)


def topk_pairwise_loss(score, target, mask=None, k=10):
    s = flatten_valid(score, mask)
    y = flatten_valid(target, mask)
    if s.numel() <= 2:
        return score.sum() * 0.0
    k = max(1, min(int(k), s.numel() // 2))
    order = torch.argsort(y, descending=True)
    top = s[order[:k]]
    bottom = s[order[-k:]]
    margin = top[:, None] - bottom[None, :]
    return F.softplus(-margin).mean()


def sigma_calibration_loss(mu, sigma, target, mask=None, eps=1e-6):
    mu_v = flatten_valid(mu, mask)
    sig_v = flatten_valid(sigma, mask).clamp_min(eps)
    y_v = flatten_valid(target, mask)
    if mu_v.numel() == 0:
        return mu.sum() * 0.0
    scaled = torch.abs(y_v - mu_v) / sig_v
    return (scaled.mean() - 1.0).abs()


def gate_balance_loss(pi):
    target = torch.ones_like(pi) / float(pi.numel())
    return torch.sum((pi - target) ** 2)


def gate_confidence_loss(pi):
    return -(pi * (pi + 1e-8).log()).sum()


def total_rcls_loss(outputs, target, mask, cfg: RCLSLossConfig, rank_target=None, point_target=None):
    score = outputs["score"]
    mu = outputs["mu"]
    sigma = outputs["sigma"]
    pi = outputs["pi"]
    rank_target = target if rank_target is None else rank_target
    point_target = target if point_target is None else point_target

    huber = masked_huber_loss(mu, point_target, mask, delta=cfg.huber_delta)
    if cfg.rank_loss == "listmle":
        rank = listmle_loss(score, rank_target, mask)
    elif cfg.rank_loss == "topk_pairwise":
        rank = topk_pairwise_loss(score, rank_target, mask, k=cfg.topk_loss_k)
    elif cfg.rank_loss == "mixed":
        rank = 0.50 * listmle_loss(score, rank_target, mask) + 0.50 * listnet_loss(
            score,
            rank_target,
            mask,
            temperature=cfg.listnet_temperature,
        )
    else:
        rank = listnet_loss(score, rank_target, mask, temperature=cfg.listnet_temperature)
    ic = negative_pearson_loss(score, rank_target, mask, eps=cfg.eps)
    topk = topk_pairwise_loss(score, rank_target, mask, k=cfg.topk_loss_k)
    if cfg.use_uncertainty:
        nll = laplace_nll(mu, sigma, point_target, mask, eps=cfg.eps)
        calib = sigma_calibration_loss(mu, sigma, point_target, mask, eps=cfg.eps)
    else:
        nll = huber.new_tensor(0.0)
        calib = huber.new_tensor(0.0)
    bal = gate_balance_loss(pi)
    conf = gate_confidence_loss(pi)
    proto = outputs.get("prototype_diversity", huber.new_tensor(0.0))
    loss = (
        cfg.loss_huber_weight * huber
        + cfg.loss_listwise_weight * rank
        + cfg.loss_nll_weight * nll
        + cfg.loss_ic_weight * ic
        + cfg.loss_topk_weight * topk
        + cfg.loss_calibration_weight * calib
        + cfg.loss_proto_diversity_weight * proto
        + cfg.loss_gate_balance_weight * bal
        + cfg.loss_gate_confidence_weight * conf
    )
    return loss, {
        "loss": loss.detach(),
        "huber": huber.detach(),
        "rank": rank.detach(),
        "ic": ic.detach(),
        "topk": topk.detach(),
        "nll": nll.detach(),
        "calibration": calib.detach(),
        "proto_diversity": proto.detach(),
        "gate_balance": bal.detach(),
        "gate_confidence": conf.detach(),
    }
