import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import (
    MARKET_CONFIGS,
    REGIME_FEATURE_NAMES,
    compute_offsets,
    get_batch,
    load_market_data,
    lookback_regime_feature_dict,
    set_seeds,
    target_day,
)
from losses_rcls import RCLSLossConfig, total_rcls_loss
from metrics import compute_metrics_from_frame
from model_rcls_proto import RCLSProtoConfig, RCLSProtoRank
from regime import (
    MANUAL_REGIME_MODES,
    REGIME_DIAGNOSTIC_COLUMNS,
    REGIME_MODES,
    SCORE_COLUMNS,
    PSEUDO_REGIME_MODES,
    build_regime_table,
    gate_summary,
    gate_target_from_row,
    make_regime_lookup,
    manual_gate_from_row,
    regime_table_to_json_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_LOG_ROOT = PROJECT_ROOT / "logs"

MODEL_VARIANTS = [
    "rcls_proto_k1",
    "rcls_proto_k2",
    "rcls_proto_k2_static_proto",
    "rcls_proto_k2_uniform_gate",
    "rcls_proto_k2_no_uncert",
    "rcls_proto_k3",
    "rcls_proto_v2",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train independent RCLS-ProtoRank.")
    parser.add_argument("--dataset", default="NASDAQ", choices=["NASDAQ", "SP500"])
    parser.add_argument("--dataset-root", default="../dataset")
    parser.add_argument("--model", default="rcls_proto_k2", choices=MODEL_VARIANTS)
    parser.add_argument("--output-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--log-root", default=str(DEFAULT_LOG_ROOT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lookback", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-regimes", type=int, default=2)
    parser.add_argument("--num-prototypes", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--prototype-delta-scale", type=float, default=0.10)
    parser.add_argument("--temperature", type=float, default=0.70)
    parser.add_argument("--lambda-uncertainty", type=float, default=0.50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--loss-huber-weight", type=float, default=1.0)
    parser.add_argument("--loss-listwise-weight", type=float, default=0.20)
    parser.add_argument("--loss-nll-weight", type=float, default=0.05)
    parser.add_argument("--loss-ic-weight", type=float, default=None)
    parser.add_argument("--loss-topk-weight", type=float, default=None)
    parser.add_argument("--loss-calibration-weight", type=float, default=None)
    parser.add_argument("--loss-proto-diversity-weight", type=float, default=None)
    parser.add_argument("--loss-gate-balance-weight", type=float, default=0.001)
    parser.add_argument("--loss-gate-confidence-weight", type=float, default=0.0005)
    parser.add_argument("--rank-loss", default="listnet", choices=["listnet", "listmle", "topk_pairwise", "mixed"])
    parser.add_argument("--listnet-temperature", type=float, default=1.0)
    parser.add_argument("--topk-loss-k", type=int, default=10)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--require-gpu", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--target-transform", default="market_residual", choices=["raw", "market_residual", "day_zscore"])
    parser.add_argument("--aux-stress-mode", default="jump_stress2", choices=["none", "pseudo_stress2", "jump_stress2"])
    parser.add_argument("--valid-selection-metric", default="composite", choices=["composite", "rankic", "ic", "p10", "loss"])
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--phase-a-epochs", type=int, default=2)
    parser.add_argument("--phase-b-epochs", type=int, default=10)
    parser.add_argument("--disable-uncertainty", action="store_true")
    parser.add_argument("--uniform-gate", action="store_true")
    parser.add_argument("--static-proto", action="store_true")
    parser.add_argument("--regime-mode", default="latent_current", choices=REGIME_MODES)
    parser.add_argument("--regime-supervision-weight", type=float, default=0.05)
    parser.add_argument("--regime-temperature", type=float, default=0.5)
    parser.add_argument("--gate-entropy-target", default="none", choices=["none", "low", "medium"])
    parser.add_argument("--gate-persistence-weight", type=float, default=None)
    parser.add_argument("--regime-jump-min-run", type=int, default=3)
    parser.add_argument("--save-prediction-splits", default="train,valid,test")
    return parser.parse_args(argv)


def resolve_dataset_root(dataset_root):
    path = Path(dataset_root)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append((Path.cwd() / path).resolve())
        candidates.append((PROJECT_ROOT / path).resolve())
        candidates.append((PROJECT_ROOT.parent / "dataset").resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def apply_variant(args):
    args.static_proto = bool(args.static_proto)
    args.uniform_gate = bool(args.uniform_gate)
    args.use_uncertainty = True

    if args.model == "rcls_proto_k1":
        args.num_regimes = 1
    elif args.model.startswith("rcls_proto_k2"):
        args.num_regimes = 2
    elif args.model == "rcls_proto_k3":
        args.num_regimes = 3
    elif args.model == "rcls_proto_v2":
        args.num_regimes = 3
        if args.regime_mode == "latent_current":
            args.regime_mode = "pseudo_market3"

    if args.model == "rcls_proto_k2_static_proto":
        args.static_proto = True
    if args.model == "rcls_proto_k2_uniform_gate":
        args.uniform_gate = True
    if args.model == "rcls_proto_k2_no_uncert":
        args.use_uncertainty = False
        args.loss_nll_weight = 0.0
    if args.disable_uncertainty:
        args.use_uncertainty = False
        args.loss_nll_weight = 0.0
        args.loss_calibration_weight = 0.0
    if args.regime_mode == "pseudo_market3":
        args.num_regimes = 3
    elif args.regime_mode != "latent_current":
        args.num_regimes = 2

    if args.model == "rcls_proto_v2":
        args.rank_loss = "mixed" if args.rank_loss == "listnet" else args.rank_loss
        if args.loss_ic_weight is None:
            args.loss_ic_weight = 0.35
        if args.loss_topk_weight is None:
            args.loss_topk_weight = 0.25
        if args.loss_calibration_weight is None:
            args.loss_calibration_weight = 0.01 if args.use_uncertainty else 0.0
        if args.loss_proto_diversity_weight is None:
            args.loss_proto_diversity_weight = 1e-4
        if args.gate_entropy_target == "none" and not args.uniform_gate:
            args.gate_entropy_target = "medium"
        if args.aux_stress_mode == "none":
            args.aux_stress_mode = "none"
    else:
        args.loss_ic_weight = 0.0 if args.loss_ic_weight is None else args.loss_ic_weight
        args.loss_topk_weight = 0.0 if args.loss_topk_weight is None else args.loss_topk_weight
        args.loss_calibration_weight = (
            0.0 if args.loss_calibration_weight is None else args.loss_calibration_weight
        )
        args.loss_proto_diversity_weight = (
            0.0 if args.loss_proto_diversity_weight is None else args.loss_proto_diversity_weight
        )
        args.ema_decay = 0.0

    if args.num_regimes == 1 or args.uniform_gate:
        args.loss_gate_balance_weight = 0.0
        args.loss_gate_confidence_weight = 0.0
    if args.regime_mode in MANUAL_REGIME_MODES:
        args.loss_gate_balance_weight = 0.0
        args.loss_gate_confidence_weight = 0.0
    if args.gate_persistence_weight is None:
        args.gate_persistence_weight = 0.001 if args.regime_mode in PSEUDO_REGIME_MODES else 0.0
    return args


def get_device(require_gpu):
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        if require_gpu and require_gpu.lower() not in name.lower():
            raise RuntimeError(f"Required GPU '{require_gpu}' not found; visible CUDA device is '{name}'.")
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda"), name
    if require_gpu:
        raise RuntimeError(f"Required GPU '{require_gpu}' but CUDA is not available.")
    return torch.device("cpu"), "cpu"


def build_model_config(args, n_features):
    return RCLSProtoConfig(
        n_features=n_features,
        lookback=args.lookback,
        d_model=args.d_model,
        num_regimes=args.num_regimes,
        num_prototypes=args.num_prototypes,
        dropout=args.dropout,
        alpha=args.alpha,
        prototype_delta_scale=args.prototype_delta_scale,
        temperature=args.temperature,
        lambda_uncertainty=args.lambda_uncertainty,
        use_uncertainty=args.use_uncertainty,
        static_proto=args.static_proto,
        uniform_gate=args.uniform_gate,
        architecture="v2" if args.model == "rcls_proto_v2" else "v1",
        decouple_rank_score=args.model == "rcls_proto_v2",
        use_stress_aux=args.model == "rcls_proto_v2" and args.aux_stress_mode != "none",
    )


def build_loss_config(args):
    return RCLSLossConfig(
        loss_huber_weight=args.loss_huber_weight,
        loss_listwise_weight=args.loss_listwise_weight,
        loss_nll_weight=args.loss_nll_weight,
        loss_ic_weight=args.loss_ic_weight,
        loss_topk_weight=args.loss_topk_weight,
        loss_calibration_weight=args.loss_calibration_weight,
        loss_proto_diversity_weight=args.loss_proto_diversity_weight,
        loss_gate_balance_weight=args.loss_gate_balance_weight,
        loss_gate_confidence_weight=args.loss_gate_confidence_weight,
        rank_loss=args.rank_loss,
        listnet_temperature=args.listnet_temperature,
        topk_loss_k=args.topk_loss_k,
        huber_delta=args.huber_delta,
        use_uncertainty=args.use_uncertainty,
    )


def selected_config_summary(args, dataset_root, device_name, split_sizes=None):
    split_sizes = split_sizes or {}
    return {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_root": str(dataset_root),
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "lookback": args.lookback,
        "steps": args.steps,
        "d_model": args.d_model,
        "num_regimes": args.num_regimes,
        "num_prototypes": args.num_prototypes,
        "static_proto": bool(args.static_proto),
        "uniform_gate": bool(args.uniform_gate),
        "use_uncertainty": bool(args.use_uncertainty),
        "architecture": "v2" if args.model == "rcls_proto_v2" else "v1",
        "run_tag": args.run_tag,
        "target_transform": args.target_transform,
        "rank_loss": args.rank_loss,
        "valid_selection_metric": args.valid_selection_metric,
        "ema_decay": args.ema_decay,
        "phase_a_epochs": args.phase_a_epochs,
        "phase_b_epochs": args.phase_b_epochs,
        "aux_stress_mode": args.aux_stress_mode,
        "dropout": args.dropout,
        "alpha": args.alpha,
        "prototype_delta_scale": args.prototype_delta_scale,
        "temperature": args.temperature,
        "lambda_uncertainty": args.lambda_uncertainty,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "loss_huber_weight": args.loss_huber_weight,
        "loss_listwise_weight": args.loss_listwise_weight,
        "loss_nll_weight": args.loss_nll_weight,
        "loss_ic_weight": args.loss_ic_weight,
        "loss_topk_weight": args.loss_topk_weight,
        "loss_calibration_weight": args.loss_calibration_weight,
        "loss_proto_diversity_weight": args.loss_proto_diversity_weight,
        "loss_gate_balance_weight": args.loss_gate_balance_weight,
        "loss_gate_confidence_weight": args.loss_gate_confidence_weight,
        "listnet_temperature": args.listnet_temperature,
        "huber_delta": args.huber_delta,
        "regime_mode": args.regime_mode,
        "regime_supervision_weight": args.regime_supervision_weight,
        "regime_temperature": args.regime_temperature,
        "gate_entropy_target": args.gate_entropy_target,
        "gate_persistence_weight": args.gate_persistence_weight,
        "regime_jump_min_run": args.regime_jump_min_run,
        "save_prediction_splits": args.save_prediction_splits,
        "device": device_name,
        "num_train_days": int(split_sizes.get("train", 0)),
        "num_valid_days": int(split_sizes.get("valid", 0)),
        "num_test_days": int(split_sizes.get("test", 0)),
    }


def tensor_batch(eod_data, mask_data, gt_data, price_data, offset, args, device):
    x, mask, _, target = get_batch(
        eod_data,
        mask_data,
        gt_data,
        price_data,
        int(offset),
        args.lookback,
        args.steps,
    )
    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(mask).to(device),
        torch.from_numpy(target.reshape(-1)).to(device),
        mask,
        target.reshape(-1),
    )


def transform_rank_target_torch(target, mask, mode, eps=1e-6):
    if mode == "raw":
        return target
    valid = mask.view(-1) > 0.5
    if not torch.any(valid):
        return target
    out = target.clone()
    values = target[valid]
    mean = values.mean()
    out = out - mean
    if mode == "day_zscore":
        std = values.std(unbiased=False).clamp_min(eps)
        out = out / std
    return out


def transform_rank_target_np(target, mask, mode, eps=1e-6):
    values = np.asarray(target, dtype=float).reshape(-1)
    if mode == "raw":
        return values.copy()
    out = values.copy()
    valid = np.asarray(mask).reshape(-1) > 0.5
    if not np.any(valid):
        return out
    mean = float(np.nanmean(out[valid]))
    out = out - mean
    if mode == "day_zscore":
        std = float(np.nanstd(values[valid]))
        out = out / max(std, eps)
    return out


def effective_alpha(args, epoch):
    if args.model != "rcls_proto_v2":
        return args.alpha
    if epoch <= max(0, args.phase_a_epochs):
        return 0.0
    if epoch <= max(args.phase_a_epochs, args.phase_b_epochs):
        denom = max(1, args.phase_b_epochs - args.phase_a_epochs)
        return args.alpha * float(epoch - args.phase_a_epochs) / float(denom)
    return args.alpha


def set_epoch_lr(optimizer, args, epoch):
    lr = args.lr
    if args.model == "rcls_proto_v2" and epoch > args.phase_b_epochs:
        lr = args.lr * 0.50
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def clone_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def init_ema_state(model):
    return clone_state_dict(model)


def update_ema_state(ema_state, model, decay):
    if ema_state is None or decay <= 0.0:
        return ema_state
    with torch.no_grad():
        for key, value in model.state_dict().items():
            value_cpu = value.detach().cpu()
            if torch.is_floating_point(value_cpu):
                ema_state[key].mul_(decay).add_(value_cpu, alpha=1.0 - decay)
            else:
                ema_state[key].copy_(value_cpu)
    return ema_state


def evaluate_with_state(model, state, fn, device):
    if state is None:
        return fn()
    live_state = clone_state_dict(model)
    model.load_state_dict({key: value.to(device) for key, value in state.items()})
    try:
        return fn()
    finally:
        model.load_state_dict({key: value.to(device) for key, value in live_state.items()})


def regime_row_for_offset(regime_lookup, offset):
    if regime_lookup is None:
        return None
    return regime_lookup.get(int(offset))


def manual_pi_for_offset(regime_lookup, offset, args, device):
    row = regime_row_for_offset(regime_lookup, offset)
    pi = manual_gate_from_row(row, args.regime_mode, args.num_regimes, args.regime_temperature)
    if pi is None:
        return None
    return pi.to(device)


def forward_with_regime(model, x_t, mask_t, offset, args, regime_lookup, device, alpha_override=None):
    manual_pi = manual_pi_for_offset(regime_lookup, offset, args, device)
    return model(x_t, mask_t, manual_pi=manual_pi, alpha_override=alpha_override)


def gate_auxiliary_loss(outputs, offset, args, regime_lookup, prev_pi=None, aux_regime_lookup=None):
    device = outputs["score"].device
    row = regime_row_for_offset(regime_lookup, offset)
    aux_row = regime_row_for_offset(aux_regime_lookup, offset)
    gate_ce = outputs["score"].sum() * 0.0
    aux_stress_ce = outputs["score"].sum() * 0.0
    gate_persistence = outputs["score"].sum() * 0.0
    gate_entropy_penalty = outputs["score"].sum() * 0.0
    if args.regime_mode in PSEUDO_REGIME_MODES and args.regime_supervision_weight > 0:
        target = gate_target_from_row(row, args.regime_mode, device)
        if target is not None:
            gate_ce = F.cross_entropy(outputs["gate_logits"].view(1, -1), target)
    if outputs.get("aux_stress_logits") is not None and aux_row is not None:
        aux_target = gate_target_from_row(aux_row, args.aux_stress_mode, device)
        if aux_target is not None:
            aux_stress_ce = F.cross_entropy(outputs["aux_stress_logits"].view(1, -1), aux_target)
    if prev_pi is not None and args.gate_persistence_weight > 0:
        gate_persistence = F.mse_loss(outputs["pi"], prev_pi.detach())
    if args.gate_entropy_target != "none":
        target_entropy = 0.25 if args.gate_entropy_target == "low" else math.log(max(args.num_regimes, 2)) * 0.70
        gate_entropy_penalty = (outputs["gate_entropy"] - target_entropy) ** 2
    total = (
        args.regime_supervision_weight * gate_ce
        + 0.50 * args.regime_supervision_weight * aux_stress_ce
        + args.gate_persistence_weight * gate_persistence
        + 0.001 * gate_entropy_penalty
    )
    return total, {
        "gate_ce": gate_ce.detach(),
        "aux_stress_ce": aux_stress_ce.detach(),
        "gate_persistence": gate_persistence.detach(),
        "gate_entropy_penalty": gate_entropy_penalty.detach(),
    }


def daily_nll(mu, sigma, target, mask):
    sig = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    y = np.asarray(target, dtype=float)
    pred_mu = np.asarray(mu, dtype=float)
    nll = np.abs(y - pred_mu) / sig + np.log(sig)
    nll[np.asarray(mask).reshape(-1) <= 0.5] = np.nan
    return nll


def collect_prediction_rows(
    model,
    split,
    offsets,
    eod_data,
    mask_data,
    gt_data,
    price_data,
    args,
    device,
    regime_lookup=None,
    aux_regime_lookup=None,
):
    rows = []
    pi_rows = []
    model.eval()
    with torch.no_grad():
        for offset in offsets:
            x_t, mask_t, _, mask_np, target_np = tensor_batch(
                eod_data,
                mask_data,
                gt_data,
                price_data,
                offset,
                args,
                device,
            )
            outputs = forward_with_regime(model, x_t, mask_t, offset, args, regime_lookup, device)
            pred = outputs["score"].detach().cpu().numpy().reshape(-1)
            rank_score = outputs.get("rank_score", outputs["score"]).detach().cpu().numpy().reshape(-1)
            mu = outputs["mu"].detach().cpu().numpy().reshape(-1)
            sigma = outputs["sigma"].detach().cpu().numpy().reshape(-1)
            pi = outputs["pi"].detach().cpu().numpy().reshape(-1)
            gate_entropy = float(outputs["gate_entropy"].detach().cpu())
            proto_delta_norm = float(outputs["proto_delta_norm"].detach().cpu())
            delta_norm = float(outputs["delta_norm"].detach().cpu())
            day_idx = target_day(int(offset), args.lookback, args.steps)
            feature_dict = lookback_regime_feature_dict(
                price_data[:, int(offset) : int(offset) + args.lookback],
                mask_np,
            )
            regime_row = regime_row_for_offset(regime_lookup, offset) or {}
            aux_row = regime_row_for_offset(aux_regime_lookup, offset) or {}
            rank_target_np = transform_rank_target_np(target_np, mask_np, args.target_transform)
            regime_diag = {
                "regime_mode": args.regime_mode,
                "regime_label": int(regime_row.get("regime_label", 0)),
                "regime_confidence": float(regime_row.get("regime_confidence", 0.0)),
                "regime_margin": float(regime_row.get("regime_margin", 0.0)),
                "aux_stress_mode": args.aux_stress_mode,
                "aux_stress_label": int(aux_row.get("regime_label", 0)),
                "aux_stress_confidence": float(aux_row.get("regime_confidence", 0.0)),
                "breadth_score": float(regime_row.get("breadth_score", np.nan)),
                "stress_score": float(regime_row.get("stress_score", np.nan)),
                "corr_score": float(regime_row.get("corr_score", np.nan)),
            }
            nll = daily_nll(mu, sigma, target_np, mask_np)
            pi_fixed = np.zeros(3, dtype=float)
            pi_fixed[: min(len(pi), 3)] = pi[:3]
            pi_rows.append(
                {
                    "dataset": args.dataset,
                    "model": args.model,
                    "run_tag": args.run_tag,
                    "seed": args.seed,
                    "split": split,
                    "day_idx": day_idx,
                    "gate_entropy": gate_entropy,
                    "proto_delta_norm": proto_delta_norm,
                    "delta_norm": delta_norm,
                    **regime_diag,
                    **{f"regime_{i}": pi_fixed[i] for i in range(3)},
                    **feature_dict,
                }
            )
            for stock_idx in range(pred.shape[0]):
                row = {
                    "dataset": args.dataset,
                    "model": args.model,
                    "run_tag": args.run_tag,
                    "seed": args.seed,
                    "split": split,
                    "day_idx": day_idx,
                    "stock_idx": stock_idx,
                    "pred": float(pred[stock_idx]),
                    "score": float(pred[stock_idx]),
                    "rank_score": float(rank_score[stock_idx]),
                    "mu": float(mu[stock_idx]),
                    "sigma": float(sigma[stock_idx]),
                    "target": float(target_np[stock_idx]),
                    "rank_target": float(rank_target_np[stock_idx]),
                    "mask": float(mask_np.reshape(-1)[stock_idx]),
                    "nll": float(nll[stock_idx]) if np.isfinite(nll[stock_idx]) else np.nan,
                    "abs_error_mu": float(abs(mu[stock_idx] - target_np[stock_idx])),
                    "sigma_1x_cover": float(abs(mu[stock_idx] - target_np[stock_idx]) <= sigma[stock_idx]),
                    "sigma_2x_cover": float(abs(mu[stock_idx] - target_np[stock_idx]) <= 2.0 * sigma[stock_idx]),
                    "gate_entropy": gate_entropy,
                    "proto_delta_norm": proto_delta_norm,
                    "delta_norm": delta_norm,
                    **regime_diag,
                }
                for idx in range(3):
                    row[f"regime_{idx}"] = float(pi_fixed[idx])
                row.update(feature_dict)
                rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(pi_rows)


def evaluate_split(
    model,
    split,
    offsets,
    eod_data,
    mask_data,
    gt_data,
    price_data,
    args,
    device,
    regime_lookup=None,
    aux_regime_lookup=None,
):
    df, gate_df = collect_prediction_rows(
        model,
        split,
        offsets,
        eod_data,
        mask_data,
        gt_data,
        price_data,
        args,
        device,
        regime_lookup=regime_lookup,
        aux_regime_lookup=aux_regime_lookup,
    )
    metrics = compute_metrics_from_frame(df)
    gate_diag = {}
    if not gate_df.empty:
        gate_diag = {
            "gate_entropy_mean": float(gate_df["gate_entropy"].mean()),
            "gate_entropy_std": float(gate_df["gate_entropy"].std(ddof=0)),
            "proto_delta_norm_mean": float(gate_df["proto_delta_norm"].mean()),
            "delta_norm_mean": float(gate_df["delta_norm"].mean()),
        }
        for idx in range(3):
            gate_diag[f"regime_{idx}_mean"] = float(gate_df[f"regime_{idx}"].mean())
        if "regime_label" in gate_df.columns:
            labels = gate_df.sort_values("day_idx")["regime_label"].astype(int).values
            switches = int(np.sum(labels[1:] != labels[:-1])) if len(labels) > 1 else 0
            occupancies = gate_df["regime_label"].value_counts(normalize=True)
            gate_diag["gate_switch_count"] = switches
            gate_diag["gate_occupancy_min"] = float(occupancies.min()) if len(occupancies) else 0.0
            gate_diag["gate_occupancy_max"] = float(occupancies.max()) if len(occupancies) else 0.0
    return df, gate_df, metrics, gate_diag


def score_for_selection(metrics, mode="rankic"):
    if mode == "composite":
        ic = metrics.get("ic", np.nan)
        p10 = metrics.get("p10", np.nan)
        p20 = metrics.get("p20", np.nan)
        ls = metrics.get("long_short", np.nan)
        values = [ic, p10, p20, ls]
        if not all(np.isfinite(x) for x in values):
            return -math.inf
        return float(
            0.35 * ((ic - 0.030) / 0.020)
            + 0.35 * ((p10 - 0.490) / 0.020)
            + 0.15 * ((p20 - 0.490) / 0.020)
            + 0.15 * ((ls - 0.0020) / 0.0030)
        )
    if mode == "ic":
        value = metrics.get("ic", np.nan)
        return float(value) if np.isfinite(value) else -math.inf
    if mode == "p10":
        value = metrics.get("p10", np.nan)
        return float(value) if np.isfinite(value) else -math.inf
    if mode == "loss":
        nll = metrics.get("nll", np.nan)
        return -float(nll) if np.isfinite(nll) else -math.inf
    rankic = metrics.get("rankic", np.nan)
    if rankic is not None and np.isfinite(rankic):
        return float(rankic)
    nll = metrics.get("nll", np.nan)
    if nll is not None and np.isfinite(nll):
        return -float(nll)
    ic = metrics.get("ic", np.nan)
    if ic is not None and np.isfinite(ic):
        return float(ic)
    return -math.inf


def mean_train_step_stats(stats):
    if not stats:
        return {}
    keys = sorted({key for row in stats for key in row})
    out = {}
    for key in keys:
        values = [row[key] for row in stats if key in row and np.isfinite(row[key])]
        out[f"train_{key}"] = float(np.mean(values)) if values else np.nan
    return out


def write_metadata(path, metadata):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def write_summary_csv(path, row):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def train(args):
    args = apply_variant(args)
    set_seeds(args.seed)
    device, device_name = get_device(args.require_gpu)
    dataset_root = resolve_dataset_root(args.dataset_root)
    output_root = Path(args.output_root).resolve()
    log_root = Path(args.log_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    eod_data, mask_data, gt_data, price_data = load_market_data(dataset_root, args.dataset, args.steps)
    market_cfg = MARKET_CONFIGS[args.dataset]
    offsets = compute_offsets(
        eod_data.shape[1],
        market_cfg["valid_index"],
        market_cfg["test_index"],
        args.lookback,
        args.steps,
    )
    regime_table, regime_artifact = build_regime_table(
        price_data,
        mask_data,
        offsets,
        args.lookback,
        args.steps,
        args.regime_mode,
        seed=args.seed,
        jump_min_run=args.regime_jump_min_run,
    )
    regime_lookup = make_regime_lookup(regime_table)
    aux_regime_table = None
    aux_regime_artifact = None
    aux_regime_lookup = None
    if args.model == "rcls_proto_v2" and args.aux_stress_mode != "none":
        aux_regime_table, aux_regime_artifact = build_regime_table(
            price_data,
            mask_data,
            offsets,
            args.lookback,
            args.steps,
            args.aux_stress_mode,
            seed=args.seed,
            jump_min_run=args.regime_jump_min_run,
        )
        aux_regime_lookup = make_regime_lookup(aux_regime_table)
    split_sizes = {key: len(value) for key, value in offsets.items()}
    summary = selected_config_summary(args, dataset_root, device_name, split_sizes)
    summary.update(gate_summary(regime_table))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.dry_run:
        return summary

    model_cfg = build_model_config(args, n_features=eod_data.shape[-1])
    model = RCLSProtoRank(model_cfg).to(device)
    loss_cfg = build_loss_config(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ema_state = init_ema_state(model) if args.ema_decay > 0.0 else None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()
    best_score = -math.inf
    best_epoch = 0
    best_state = None
    best_valid_metrics = None
    best_ic_epoch = 0
    best_ic_state = None
    best_ic_value = -math.inf
    best_p10_epoch = 0
    best_p10_state = None
    best_p10_value = -math.inf
    stale_epochs = 0
    epoch_rows = []

    for epoch in range(1, args.epochs + 1):
        epoch_lr = set_epoch_lr(optimizer, args, epoch)
        alpha_now = effective_alpha(args, epoch)
        model.train()
        epoch_stats = []
        train_offsets = np.asarray(offsets["train"], dtype=int).copy()
        if args.gate_persistence_weight > 0:
            train_offsets.sort()
        else:
            np.random.shuffle(train_offsets)
        prev_pi = None
        for offset in train_offsets:
            x_t, mask_t, target_t, _, _ = tensor_batch(
                eod_data,
                mask_data,
                gt_data,
                price_data,
                offset,
                args,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            rank_target_t = transform_rank_target_torch(target_t, mask_t, args.target_transform)
            outputs = forward_with_regime(
                model,
                x_t,
                mask_t,
                offset,
                args,
                regime_lookup,
                device,
                alpha_override=alpha_now,
            )
            loss, parts = total_rcls_loss(
                outputs,
                target_t,
                mask_t,
                loss_cfg,
                rank_target=rank_target_t,
                point_target=target_t,
            )
            gate_aux, gate_parts = gate_auxiliary_loss(
                outputs,
                offset,
                args,
                regime_lookup,
                prev_pi,
                aux_regime_lookup=aux_regime_lookup,
            )
            loss = loss + gate_aux
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            ema_state = update_ema_state(ema_state, model, args.ema_decay)
            stat = {key: float(value.detach().cpu()) for key, value in parts.items()}
            stat["loss"] = float(loss.detach().cpu())
            for key, value in gate_parts.items():
                stat[key] = float(value.detach().cpu())
            stat["gate_entropy"] = float(outputs["gate_entropy"].detach().cpu())
            stat["proto_delta_norm"] = float(outputs["proto_delta_norm"].detach().cpu())
            stat["delta_norm"] = float(outputs["delta_norm"].detach().cpu())
            regime_row = regime_row_for_offset(regime_lookup, offset) or {}
            for key in REGIME_DIAGNOSTIC_COLUMNS:
                if key in regime_row:
                    stat[key] = float(regime_row[key])
            for idx, value in enumerate(outputs["pi"].detach().cpu().numpy().reshape(-1)):
                stat[f"regime_{idx}"] = float(value)
            prev_pi = outputs["pi"].detach()
            epoch_stats.append(stat)

        _, _, valid_metrics, valid_gate_diag = evaluate_with_state(
            model,
            ema_state,
            lambda: evaluate_split(
                model,
                "valid",
                offsets["valid"],
                eod_data,
                mask_data,
                gt_data,
                price_data,
                args,
                device,
                regime_lookup=regime_lookup,
                aux_regime_lookup=aux_regime_lookup,
            ),
            device,
        )
        select_score = score_for_selection(valid_metrics, args.valid_selection_metric)
        improved = select_score > best_score
        if improved:
            best_score = select_score
            best_epoch = epoch
            stale_epochs = 0
            best_valid_metrics = dict(valid_metrics)
            best_state = {key: value.detach().cpu().clone() for key, value in (ema_state or model.state_dict()).items()}
        else:
            stale_epochs += 1
        valid_ic = valid_metrics.get("ic", np.nan)
        if np.isfinite(valid_ic) and valid_ic > best_ic_value:
            best_ic_value = float(valid_ic)
            best_ic_epoch = epoch
            best_ic_state = {key: value.detach().cpu().clone() for key, value in (ema_state or model.state_dict()).items()}
        valid_p10 = valid_metrics.get("p10", np.nan)
        if np.isfinite(valid_p10) and valid_p10 > best_p10_value:
            best_p10_value = float(valid_p10)
            best_p10_epoch = epoch
            best_p10_state = {key: value.detach().cpu().clone() for key, value in (ema_state or model.state_dict()).items()}

        epoch_row = {
            "dataset": args.dataset,
            "model": args.model,
            "seed": args.seed,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "selection_score": select_score,
            "epoch_lr": epoch_lr,
            "alpha_effective": alpha_now,
            **mean_train_step_stats(epoch_stats),
        }
        for key, value in valid_metrics.items():
            epoch_row[f"valid_{key}"] = value
        epoch_row.update(valid_gate_diag)
        epoch_rows.append(epoch_row)
        print(
            f"epoch={epoch:03d} train_loss={epoch_row.get('train_loss', np.nan):.6f} "
            f"valid_rankic={valid_metrics.get('rankic', np.nan):.6f} "
            f"valid_ic={valid_metrics.get('ic', np.nan):.6f} best_epoch={best_epoch}"
        )
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; no validation improvement for {args.patience} epochs.")
            break

    if best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})

    pred_frames = []
    gate_frames = []
    final_metrics = {}
    save_splits = {x.strip() for x in args.save_prediction_splits.split(",") if x.strip()}
    if "all" in save_splits:
        save_splits = {"train", "valid", "test"}
    for split in ["train", "valid", "test"]:
        pred_df, gate_df, metrics, _ = evaluate_split(
            model,
            split,
            offsets[split],
            eod_data,
            mask_data,
            gt_data,
            price_data,
            args,
            device,
            regime_lookup=regime_lookup,
            aux_regime_lookup=aux_regime_lookup,
        )
        if split in save_splits:
            pred_frames.append(pred_df)
        gate_frames.append(gate_df)
        final_metrics[split] = metrics

    all_pred = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    all_gate = pd.concat(gate_frames, ignore_index=True)
    tag = f"_{args.run_tag}" if args.run_tag else ""
    stem = f"{args.model}_{args.regime_mode}{tag}_{args.dataset}_seed{args.seed}"
    pred_path = output_root / f"preds_{stem}.csv"
    metadata_path = output_root / f"metadata_{stem}.json"
    epoch_path = output_root / f"epoch_diagnostics_{stem}.csv"
    gate_path = output_root / f"gate_proto_diagnostics_{stem}.csv"
    regime_path = output_root / f"regime_table_{stem}.csv"
    aux_regime_path = output_root / f"aux_regime_table_{stem}.csv"
    summary_path = output_root / f"run_summary_{stem}.csv"
    checkpoint_path = output_root / f"checkpoint_{stem}.pt"
    checkpoint_ic_path = output_root / f"checkpoint_{stem}_best_ic.pt"
    checkpoint_p10_path = output_root / f"checkpoint_{stem}_best_p10.pt"

    all_pred.to_csv(pred_path, index=False)
    all_gate.to_csv(gate_path, index=False)
    regime_table.to_csv(regime_path, index=False)
    if aux_regime_table is not None:
        aux_regime_table.to_csv(aux_regime_path, index=False)
    pd.DataFrame(epoch_rows).to_csv(epoch_path, index=False)

    elapsed = time.time() - start_time
    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))
    metadata = {
        **summary,
        "num_params": int(num_params),
        "best_epoch": int(best_epoch),
        "best_ic_epoch": int(best_ic_epoch),
        "best_ic_value": best_ic_value,
        "best_p10_epoch": int(best_p10_epoch),
        "best_p10_value": best_p10_value,
        "best_selection_score": best_score,
        "best_valid_metrics": best_valid_metrics,
        "final_metrics": final_metrics,
        "prediction_file": str(pred_path),
        "gate_proto_diagnostics_file": str(gate_path),
        "regime_table_file": str(regime_path),
        "epoch_diagnostics_file": str(epoch_path),
        "elapsed_seconds": elapsed,
        "peak_vram_mb": peak_vram_mb,
        "regime_feature_names": REGIME_FEATURE_NAMES,
        "regime_artifact": regime_artifact.to_metadata(),
        "aux_regime_artifact": aux_regime_artifact.to_metadata() if aux_regime_artifact is not None else None,
        "regime_table_preview": regime_table_to_json_records(regime_table.head(20)),
    }
    if aux_regime_table is not None:
        metadata["aux_regime_table_file"] = str(aux_regime_path)
    write_metadata(metadata_path, metadata)

    summary_row = {
        "dataset": args.dataset,
        "model": args.model,
        "regime_mode": args.regime_mode,
        "run_tag": args.run_tag,
        "target_transform": args.target_transform,
        "rank_loss": args.rank_loss,
        "valid_selection_metric": args.valid_selection_metric,
        "aux_stress_mode": args.aux_stress_mode,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_ic_epoch": best_ic_epoch,
        "best_p10_epoch": best_p10_epoch,
        "elapsed_seconds": elapsed,
        "peak_vram_mb": peak_vram_mb,
        "num_params": int(num_params),
    }
    summary_row.update(gate_summary(regime_table))
    for split, metrics in final_metrics.items():
        for key, value in metrics.items():
            summary_row[f"{split}_{key}"] = value
    write_summary_csv(summary_path, summary_row)

    if args.save_checkpoint:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "best_epoch": best_epoch,
                "best_ic_epoch": best_ic_epoch,
                "best_p10_epoch": best_p10_epoch,
                "best_valid_metrics": best_valid_metrics,
                "best_selection_score": best_score,
            },
            checkpoint_path,
        )
        if best_ic_state is not None:
            torch.save(
                {
                    "model_state_dict": best_ic_state,
                    "args": vars(args),
                    "best_epoch": best_ic_epoch,
                    "best_metric": "valid_ic",
                    "best_metric_value": best_ic_value,
                },
                checkpoint_ic_path,
            )
        if best_p10_state is not None:
            torch.save(
                {
                    "model_state_dict": best_p10_state,
                    "args": vars(args),
                    "best_epoch": best_p10_epoch,
                    "best_metric": "valid_p10",
                    "best_metric_value": best_p10_value,
                },
                checkpoint_p10_path,
            )
        metadata["checkpoint_file"] = str(checkpoint_path)
        metadata["checkpoint_best_ic_file"] = str(checkpoint_ic_path)
        metadata["checkpoint_best_p10_file"] = str(checkpoint_p10_path)
        write_metadata(metadata_path, metadata)

    print(f"Wrote predictions: {pred_path}")
    print(f"Wrote metadata: {metadata_path}")
    print(f"Wrote epoch diagnostics: {epoch_path}")
    return metadata


def main(argv=None):
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
