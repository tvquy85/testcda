import argparse
import csv
import json
import pickle
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluator import evaluate
from model import StockMixer, get_loss
from model_rcls_delta import REGIME_FEATURE_NAMES
from regime_modes import (
    MANUAL_REGIME_MODES,
    NEW_REGIME_MODES,
    PSEUDO_REGIME_MODES,
    REGIME_DIAGNOSTIC_COLUMNS,
    REGIME_MODES,
    apply_regime_artifact,
    fit_regime_artifact,
    make_regime_lookup,
    manual_gate_from_row,
    regime_summary,
)


BASE_NUMPY_SEED = 123456789
BASE_TORCH_SEED = 12345678

PAPER_NASDAQ = {
    "IC": 0.043,
    "RIC": 0.501,
    "prec_10": 0.545,
    "sharpe5": 1.465,
}

ACTIVATIONS = ("hardswish", "relu", "gelu")
GATE_FEATURE_MODES = ("stress_embedding", "embedding_only", "stress_only")
MODEL_CHOICES = (
    "stockmixer",
    "stockmixer_ft",
    "rcls_f_k3",
    "rcls_f_k1",
    "rcls_delta_identity",
    "rcls_delta_k1",
    "rcls_delta_k2",
    "rcls_delta_k2_nostress",
    "rcls_delta_k2_uniform",
    "rcls_delta_k3",
    "rcls_delta_k3_uniform",
)
EPOCH_PRESETS = {
    "smoke": 1,
    "quick": 20,
    "paper": 100,
}

MARKET_CONFIGS = {
    "NASDAQ": {
        "stock_num": 1026,
        "valid_index": 756,
        "test_index": 1008,
        "market_num": 20,
    },
    "SP500": {
        "stock_num": 474,
        "valid_index": 1006,
        "test_index": 1259,
        "market_num": 8,
    },
}


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in ("true", "1", "yes", "y"):
        return True
    if lowered in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected true or false, got {}".format(value))


def is_rcls_delta(model_name):
    return model_name.startswith("rcls_delta")


def variant_num_regimes(model_name, default_num_regimes):
    if model_name == "rcls_delta_identity" or model_name == "rcls_delta_k1":
        return 1
    if "k3" in model_name:
        return 3
    if "k2" in model_name:
        return 2
    return default_num_regimes

PREDICTION_COLUMNS = [
    "dataset",
    "model",
    "seed",
    "split",
    "day_idx",
    "stock_idx",
    "pred",
    "target",
    "mask",
    "market_vol_lookback",
    "synchronism_lookback",
    "dispersion_lookback",
    "mean_abs_ret_lookback",
    "stress_source",
    "regime_mode",
    "regime_label",
    "regime_confidence",
    "regime_margin",
    "breadth_score",
    "stress_score",
    "corr_score",
    "regime_0",
    "regime_1",
    "regime_2",
    "gate_entropy",
    "dominant_regime",
    "delta_norm",
    "base_norm",
    "delta_scale",
    "pseudo_regime_label",
]

GATE_FEATURE_COLUMNS = [
    "dataset",
    "model",
    "seed",
    "day_idx",
    "split",
    "regime_mode",
    "regime_label",
    "regime_confidence",
    "regime_margin",
    "breadth_score",
    "stress_score",
    "corr_score",
    "regime_0",
    "regime_1",
    "regime_2",
    "gate_entropy",
    "dominant_regime",
    "market_ret_mean",
    "market_ret_std",
    "market_ret_last",
    "downside_vol",
    "dispersion",
    "synchronism",
    "mean_abs_ret",
    "max_abs_ret",
    "frac_positive",
    "delta_norm",
    "base_norm",
    "delta_scale",
    "pseudo_regime_label",
]

CONFIG_SUMMARY_FIELDS = [
    "model",
    "dataset",
    "num_regimes",
    "regime_mode",
    "regime_temperature",
    "regime_jump_min_run",
    "uniform_gate",
    "gate_feature_mode",
    "gate_pseudo_label",
    "gate_pseudo_weight",
    "gate_pseudo_final_weight",
    "gate_pseudo_warmup_epochs",
    "gate_confidence_weight",
    "expert_diversity_weight",
    "delta_scale",
    "delta_trainable_scale",
    "delta_dropout",
    "gate_temperature",
    "freeze_base_epochs",
    "warmstart_checkpoint",
    "save_checkpoint",
    "checkpoint_dir",
    "results_dir",
]

EPOCH_DIAGNOSTIC_COLUMNS = [
    "dataset",
    "model",
    "seed",
    "epoch",
    "regime_mode",
    "loss_main",
    "loss_gate_ce",
    "loss_gate_confidence",
    "loss_expert_diversity",
    "loss_total",
    "mean_gate_entropy",
    "std_gate_entropy",
    "mean_pi_0",
    "mean_pi_1",
    "mean_pi_2",
    "pseudo_label_counts",
]

METADATA_COLUMNS = [
    "dataset",
    "model",
    "seed",
    "epochs",
    "epochs_ran",
    "patience",
    "learning_rate",
    "batch_size",
    "num_params",
    "train_time_sec",
    "total_time_sec",
    "max_vram_gb",
    "infer_time_ms_per_day",
    "best_epoch",
    "best_valid_loss",
    "numpy_seed",
    "torch_seed",
    "git_commit",
    "device",
    "num_regimes",
    "regime_mode",
    "regime_temperature",
    "regime_jump_min_run",
    "regime_label_counts",
    "train_regime_switch_count",
    "valid_regime_switch_count",
    "test_regime_switch_count",
    "train_regime_occupancy_min",
    "valid_regime_occupancy_min",
    "test_regime_occupancy_min",
    "delta_scale",
    "delta_trainable_scale",
    "delta_dropout",
    "gate_temperature",
    "gate_feature_mode",
    "uniform_gate",
    "gate_pseudo_label",
    "gate_pseudo_weight",
    "gate_pseudo_final_weight",
    "gate_pseudo_warmup_epochs",
    "gate_confidence_weight",
    "expert_diversity_weight",
    "freeze_base_epochs",
    "warmstart_checkpoint",
    "save_checkpoint",
    "checkpoint_dir",
]


def parse_args(argv=None):
    submission_root = Path(__file__).resolve().parents[2]
    repo_root = submission_root.parent
    parser = argparse.ArgumentParser(
        description="Train StockMixer/RCLS-F variants for the submission pack.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--market", default=None, choices=sorted(MARKET_CONFIGS))
    parser.add_argument(
        "--dataset",
        default=None,
        choices=sorted(MARKET_CONFIGS),
        help="Alias for --market.",
    )
    parser.add_argument("--model", default="stockmixer", choices=MODEL_CHOICES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs to train. Overrides --epoch-preset when provided.",
    )
    parser.add_argument(
        "--epoch-preset",
        choices=sorted(EPOCH_PRESETS),
        default=None,
        help="Convenience epoch count: smoke=1, quick=20, paper=100.",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument(
        "--require-gpu",
        default="RTX 3090",
        help="Required substring in cuda:0 device name. Use empty value to allow CPU/any GPU.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "dataset",
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=submission_root,
        help="Root where logs/results/document live.",
    )
    parser.add_argument(
        "--results-dir",
        "--input-dir",
        dest="results_dir",
        type=Path,
        default=None,
        help="Directory for result CSVs. Defaults to <output-root>/results.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for saved checkpoints. Defaults to <output-root>/checkpoints.",
    )
    parser.add_argument("--save-checkpoint", type=str_to_bool, default=False)
    parser.add_argument("--dry-run", type=str_to_bool, default=False)
    parser.add_argument("--lookback-length", type=int, default=16)
    parser.add_argument("--fea-num", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--scale-factor", type=int, default=3)
    parser.add_argument("--gate-hidden", type=int, default=64)
    parser.add_argument("--rcls-dropout", type=float, default=0.10)
    parser.add_argument("--num-regimes", type=int, default=2)
    parser.add_argument("--delta-scale", type=float, default=0.05)
    parser.add_argument("--delta-trainable-scale", type=str_to_bool, default=True)
    parser.add_argument("--delta-dropout", type=float, default=0.05)
    parser.add_argument("--gate-temperature", type=float, default=0.7)
    parser.add_argument(
        "--gate-feature-mode",
        choices=GATE_FEATURE_MODES,
        default="stress_embedding",
    )
    parser.add_argument("--uniform-gate", type=str_to_bool, default=False)
    parser.add_argument("--gate-pseudo-label", type=str_to_bool, default=True)
    parser.add_argument("--gate-pseudo-weight", type=float, default=0.02)
    parser.add_argument("--gate-pseudo-warmup-epochs", type=int, default=20)
    parser.add_argument("--gate-pseudo-final-weight", type=float, default=0.005)
    parser.add_argument("--gate-confidence-weight", type=float, default=0.0005)
    parser.add_argument("--expert-diversity-weight", type=float, default=0.00001)
    parser.add_argument(
        "--regime-mode",
        choices=REGIME_MODES,
        default="legacy_delta",
        help="Regime label/gate mode. legacy_delta preserves the original RCLS-Delta behavior.",
    )
    parser.add_argument("--regime-temperature", type=float, default=0.5)
    parser.add_argument("--regime-jump-min-run", type=int, default=3)
    parser.add_argument("--freeze-base-epochs", type=int, default=0)
    parser.add_argument("--warmstart-checkpoint", default="")
    parser.add_argument(
        "--activation",
        choices=ACTIVATIONS,
        default="hardswish",
        help="Default activation for mixer branches unless overridden.",
    )
    parser.add_argument(
        "--main-mixer-activation",
        choices=ACTIVATIONS,
        default=None,
        help="Activation used in mixer.mix_layer.channelMixer.LN.",
    )
    parser.add_argument(
        "--scale-mixer-activation",
        choices=ACTIVATIONS,
        default=None,
        help="Activation used in mixer.scale_mix_layer.channelMixer.LN.",
    )
    parser.add_argument(
        "--stock-activation",
        choices=ACTIVATIONS,
        default="hardswish",
        help="Activation used in the stock mixing block.",
    )
    parser.add_argument("--numpy-seed", type=int, default=None)
    parser.add_argument("--torch-seed", type=int, default=None)
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Disable best-epoch prediction CSV writing.",
    )
    args = parser.parse_args(argv)
    return resolve_args(parser, args)


def resolve_args(parser, args):
    if args.market is None and args.dataset is None:
        args.market = "NASDAQ"
    elif args.market is None:
        args.market = args.dataset
    elif args.dataset is not None and args.dataset != args.market:
        parser.error("--dataset and --market must match when both are provided")
    args.dataset = args.market

    if args.epochs is None:
        if args.epoch_preset is None:
            args.epoch_preset = "paper"
        args.epochs = EPOCH_PRESETS[args.epoch_preset]
    elif args.epochs <= 0:
        parser.error("--epochs must be a positive integer")
    elif args.epoch_preset is None:
        args.epoch_preset = "custom"

    if args.runs <= 0:
        parser.error("--runs must be a positive integer")
    if args.patience < 0:
        parser.error("--patience must be non-negative")
    if args.gate_hidden <= 0:
        parser.error("--gate-hidden must be positive")
    if args.rcls_dropout < 0.0 or args.rcls_dropout >= 1.0:
        parser.error("--rcls-dropout must be in [0, 1)")
    if args.num_regimes <= 0:
        parser.error("--num-regimes must be positive")
    if args.delta_dropout < 0.0 or args.delta_dropout >= 1.0:
        parser.error("--delta-dropout must be in [0, 1)")
    if args.gate_temperature <= 0.0:
        parser.error("--gate-temperature must be positive")
    if args.regime_temperature <= 0.0:
        parser.error("--regime-temperature must be positive")
    if args.regime_jump_min_run <= 0:
        parser.error("--regime-jump-min-run must be positive")
    if args.gate_pseudo_warmup_epochs < 0:
        parser.error("--gate-pseudo-warmup-epochs must be non-negative")
    if args.freeze_base_epochs < 0:
        parser.error("--freeze-base-epochs must be non-negative")
    if args.main_mixer_activation is None:
        args.main_mixer_activation = args.activation
    if args.scale_mixer_activation is None:
        args.scale_mixer_activation = args.activation
    if args.numpy_seed is None:
        args.numpy_seed = BASE_NUMPY_SEED + args.seed
    if args.torch_seed is None:
        args.torch_seed = BASE_TORCH_SEED + args.seed

    if args.results_dir is None:
        args.results_dir = args.output_root / "results"
    if args.checkpoint_dir is None:
        args.checkpoint_dir = args.output_root / "checkpoints"

    if is_rcls_delta(args.model):
        args.num_regimes = variant_num_regimes(args.model, args.num_regimes)
        if args.regime_mode == "pseudo_market3" and args.num_regimes != 3:
            parser.error("--regime-mode pseudo_market3 requires an rcls_delta_k3 model")
        if args.regime_mode in NEW_REGIME_MODES and args.regime_mode != "pseudo_market3":
            if args.num_regimes != 2:
                parser.error("--regime-mode {} requires an rcls_delta_k2 model".format(args.regime_mode))
        if args.model.endswith("_uniform"):
            if args.regime_mode != "legacy_delta":
                parser.error("Non-legacy --regime-mode is incompatible with uniform-gate variants")
            args.uniform_gate = True
            args.gate_pseudo_label = False
            args.gate_pseudo_weight = 0.0
            args.gate_pseudo_final_weight = 0.0
            args.gate_confidence_weight = 0.0
            args.expert_diversity_weight = 0.0
        if args.model.endswith("_nostress"):
            args.gate_feature_mode = "embedding_only"
        if args.model == "rcls_delta_identity":
            args.num_regimes = 1
            args.delta_scale = 0.0
            args.delta_trainable_scale = False
            args.uniform_gate = True
            args.gate_pseudo_label = False
            args.gate_pseudo_weight = 0.0
            args.gate_pseudo_final_weight = 0.0
            args.gate_confidence_weight = 0.0
            args.expert_diversity_weight = 0.0
        if args.model == "rcls_delta_k1":
            args.num_regimes = 1
            if args.regime_mode != "legacy_delta":
                parser.error("Non-legacy --regime-mode requires k2 or k3, not rcls_delta_k1")
            args.gate_pseudo_label = False
            args.gate_pseudo_weight = 0.0
            args.gate_pseudo_final_weight = 0.0
            args.gate_confidence_weight = 0.0
            args.expert_diversity_weight = 0.0
        if args.regime_mode in PSEUDO_REGIME_MODES:
            args.gate_pseudo_label = True
        if args.regime_mode in MANUAL_REGIME_MODES:
            args.gate_pseudo_label = False
            args.gate_pseudo_weight = 0.0
            args.gate_pseudo_final_weight = 0.0
            args.gate_confidence_weight = 0.0
    else:
        if args.regime_mode != "legacy_delta":
            parser.error("--regime-mode {} requires an rcls_delta model".format(args.regime_mode))
        args.regime_mode = "legacy_delta"
        args.uniform_gate = False
        args.gate_pseudo_label = False
        args.gate_pseudo_weight = 0.0
        args.gate_pseudo_final_weight = 0.0
        args.gate_confidence_weight = 0.0
        args.expert_diversity_weight = 0.0
    args.save_predictions = not args.no_save_predictions
    return args


def selected_config(args):
    output = {}
    for name in CONFIG_SUMMARY_FIELDS:
        value = getattr(args, name, "")
        if isinstance(value, Path):
            value = str(value)
        output[name] = value
    return output


def serializable_args(args):
    output = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            value = str(value)
        output[key] = value
    return output


def print_selected_config(args):
    print("Selected RCLS-Delta config:")
    print(json.dumps(selected_config(args), indent=2, sort_keys=True))


def get_device(required_gpu):
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device_name = torch.cuda.get_device_name(0)
        if required_gpu and required_gpu not in device_name:
            raise RuntimeError(
                "cuda:0 is '{}', expected a GPU name containing '{}'.".format(
                    device_name, required_gpu
                )
            )
        return torch.device("cuda:0"), device_name
    if required_gpu:
        raise RuntimeError("CUDA is not available; this run requires cuda:0.")
    return torch.device("cpu"), "cpu"


def set_seeds(numpy_seed, torch_seed):
    random.seed(numpy_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)


def load_market_data(dataset_root, market_name, steps):
    dataset_root = dataset_root.resolve()
    if market_name == "SP500":
        data = np.load(dataset_root / "SP500" / "SP500.npy")
        data = data[:, 915:, :]
        price_data = data[:, :, -1]
        mask_data = np.ones((data.shape[0], data.shape[1]))
        eod_data = data
        gt_data = np.zeros((data.shape[0], data.shape[1]))
        for ticket in range(data.shape[0]):
            for row in range(1, data.shape[1]):
                gt_data[ticket][row] = (
                    data[ticket][row][-1] - data[ticket][row - steps][-1]
                ) / data[ticket][row - steps][-1]
        return eod_data, mask_data, gt_data, price_data

    dataset_path = dataset_root / market_name
    with open(dataset_path / "eod_data.pkl", "rb") as f:
        eod_data = pickle.load(f)
    with open(dataset_path / "mask_data.pkl", "rb") as f:
        mask_data = pickle.load(f)
    with open(dataset_path / "gt_data.pkl", "rb") as f:
        gt_data = pickle.load(f)
    with open(dataset_path / "price_data.pkl", "rb") as f:
        price_data = pickle.load(f)
    return eod_data, mask_data, gt_data, price_data


def format_metrics(perf):
    return (
        "mse:{mse:.2e}, IC:{IC:.2e}, RIC:{RIC:.2e}, "
        "prec@10:{prec_10:.2e}, SR:{sharpe5:.2e}"
    ).format(**perf)


def print_paper_comparison(market_name, best_test_perf):
    if market_name != "NASDAQ":
        return
    print("Paper NASDAQ StockMixer target:")
    print(
        "IC:{IC:.3f}, RIC:{RIC:.3f}, prec@10:{prec_10:.3f}, SR:{sharpe5:.3f}".format(
            **PAPER_NASDAQ
        )
    )
    print("Delta vs paper (run - paper):")
    print(
        "IC:{:+.3f}, RIC:{:+.3f}, prec@10:{:+.3f}, SR:{:+.3f}".format(
            best_test_perf["IC"] - PAPER_NASDAQ["IC"],
            best_test_perf["RIC"] - PAPER_NASDAQ["RIC"],
            best_test_perf["prec_10"] - PAPER_NASDAQ["prec_10"],
            best_test_perf["sharpe5"] - PAPER_NASDAQ["sharpe5"],
        )
    )


def git_commit():
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return ""


def upsert_csv_row(path, fieldnames, row, key_fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for existing in reader:
                same_key = all(str(existing.get(k, "")) == str(row.get(k, "")) for k in key_fields)
                if not same_key:
                    rows.append(existing)
    rows.append(row)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in rows:
            writer.writerow(item)


def safe_float(value):
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value):
        return ""
    return value


def lookback_regime_feature_dict(price_window, eps=1e-6):
    prices = np.asarray(price_window, dtype=float)
    if prices.ndim != 2 or prices.shape[1] < 2:
        return {name: np.nan for name in REGIME_FEATURE_NAMES}
    prices = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)
    prev = prices[:, :-1]
    nxt = prices[:, 1:]
    returns = (nxt - prev) / (np.abs(prev) + eps)
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    returns = np.clip(returns, -0.5, 0.5)

    market_ret_t = returns.mean(axis=0)
    downside = np.minimum(market_ret_t, 0.0)
    market_sign = np.sign(market_ret_t)[None, :]
    stock_sign = np.sign(returns)
    features = {
        "market_ret_mean": float(np.mean(market_ret_t)),
        "market_ret_std": float(np.std(market_ret_t)),
        "market_ret_last": float(market_ret_t[-1]),
        "downside_vol": float(np.sqrt(np.mean(downside ** 2) + eps)),
        "dispersion": float(np.mean(np.std(returns, axis=0))),
        "synchronism": float(np.mean(stock_sign == market_sign)),
        "mean_abs_ret": float(np.mean(np.abs(returns))),
        "max_abs_ret": float(np.max(np.abs(returns))),
        "frac_positive": float(np.mean(returns > 0)),
    }
    return features


def gate_entropy_from_probs(values):
    probs = np.asarray([x for x in values if x != ""], dtype=float)
    if probs.size == 0 or not np.isfinite(probs).any():
        return ""
    total = probs.sum()
    if total <= 0:
        return ""
    probs = np.clip(probs / total, 1e-12, 1.0)
    return safe_float(-(probs * np.log(probs)).sum())


class Trainer:
    def __init__(self, args, config, device, data, run_index, device_name):
        self.args = args
        self.config = config
        self.device = device
        self.device_name = device_name
        self.run_index = run_index
        self.result_seed = args.seed + run_index
        self.numpy_seed = args.numpy_seed + run_index
        self.torch_seed = args.torch_seed + run_index
        self.eod_data, self.mask_data, self.gt_data, self.price_data = data
        self.trade_dates = self.mask_data.shape[1]
        self.train_offset_end = (
            config["valid_index"] - args.lookback_length - args.steps + 1
        )
        if self.train_offset_end <= 0:
            raise ValueError("No valid training offsets for this configuration.")
        self.train_offsets = np.arange(start=0, stop=self.train_offset_end, dtype=int)
        self.output_root = args.output_root.resolve()
        self.results_dir = args.results_dir.resolve()
        self.checkpoint_dir = args.checkpoint_dir.resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if args.save_checkpoint:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.infer_time_ms_per_day = 0.0
        self.regime_stats = self.fit_regime_stats(self.train_offsets)
        self.regime_rows = []
        self.regime_lookup = {}
        self.regime_artifact = None
        self.regime_summary_stats = {}
        if is_rcls_delta(args.model) and args.regime_mode in NEW_REGIME_MODES:
            self.setup_new_regime_mode()

        self.model = StockMixer(
            stocks=config["stock_num"],
            time_steps=args.lookback_length,
            channels=args.fea_num,
            market=config["market_num"],
            scale=args.scale_factor,
            activation=args.activation,
            main_mixer_activation=args.main_mixer_activation,
            scale_mixer_activation=args.scale_mixer_activation,
            stock_activation=args.stock_activation,
            model_name=args.model,
            gate_hidden=args.gate_hidden,
            rcls_dropout=args.rcls_dropout,
            num_regimes=args.num_regimes,
            delta_scale=args.delta_scale,
            delta_trainable_scale=args.delta_trainable_scale,
            delta_dropout=args.delta_dropout,
            gate_temperature=args.gate_temperature,
            gate_feature_mode=args.gate_feature_mode,
            uniform_gate=args.uniform_gate,
        ).to(device)
        self.load_warmstart_checkpoint(args.warmstart_checkpoint)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.save_regime_thresholds()

    def run_stem(self):
        base = "{}_{}_seed{}".format(self.args.model, self.args.market, self.result_seed)
        if self.args.regime_mode == "legacy_delta":
            return base
        return "{}_{}_{}_seed{}".format(
            self.args.model,
            self.args.regime_mode,
            self.args.market,
            self.result_seed,
        )

    def split_offsets(self):
        valid_start = self.config["valid_index"] - self.args.lookback_length - self.args.steps + 1
        valid_end = self.config["test_index"] - self.args.lookback_length - self.args.steps + 1
        test_start = self.config["test_index"] - self.args.lookback_length - self.args.steps + 1
        test_end = self.trade_dates - self.args.lookback_length - self.args.steps + 1
        return {
            "train": self.train_offsets,
            "valid": np.arange(valid_start, valid_end, dtype=int),
            "test": np.arange(test_start, test_end, dtype=int),
        }

    def build_regime_feature_rows(self):
        rows = []
        for split, offsets in self.split_offsets().items():
            for offset in offsets:
                offset = int(offset)
                row = {
                    "split": split,
                    "offset": offset,
                    "day_idx": int(self.offset_target_day(offset)),
                }
                row.update(self.offset_regime_features(offset))
                rows.append(row)
        return rows

    def setup_new_regime_mode(self):
        feature_rows = self.build_regime_feature_rows()
        self.regime_artifact = fit_regime_artifact(
            feature_rows,
            self.args.regime_mode,
            self.args.num_regimes,
            jump_min_run=self.args.regime_jump_min_run,
        )
        self.regime_rows = apply_regime_artifact(feature_rows, self.regime_artifact)
        self.regime_lookup = make_regime_lookup(self.regime_rows)
        self.regime_summary_stats = regime_summary(self.regime_rows)

    def regime_row(self, offset):
        if not self.regime_lookup:
            return None
        return self.regime_lookup.get(int(offset))

    def regime_diagnostics(self, offset):
        row = self.regime_row(offset)
        if row is None:
            label = self.pseudo_regime_label(offset) if is_rcls_delta(self.args.model) else ""
            return {
                "regime_mode": self.args.regime_mode if is_rcls_delta(self.args.model) else "",
                "regime_label": label,
                "regime_confidence": "",
                "regime_margin": "",
                "breadth_score": "",
                "stress_score": "",
                "corr_score": "",
            }
        return {
            "regime_mode": self.args.regime_mode,
            "regime_label": int(row.get("regime_label", 0)),
            "regime_confidence": safe_float(row.get("regime_confidence")),
            "regime_margin": safe_float(row.get("regime_margin")),
            "breadth_score": safe_float(row.get("breadth_score")),
            "stress_score": safe_float(row.get("stress_score")),
            "corr_score": safe_float(row.get("corr_score")),
        }

    def manual_pi_for_offset(self, offset):
        if self.args.regime_mode not in MANUAL_REGIME_MODES:
            return None
        row = self.regime_row(offset)
        pi = manual_gate_from_row(
            row,
            self.args.regime_mode,
            self.args.num_regimes,
            temperature=self.args.regime_temperature,
        )
        if pi is None:
            return None
        return pi.to(self.device)

    def load_warmstart_checkpoint(self, checkpoint):
        if not checkpoint:
            return
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError("Warmstart checkpoint not found: {}".format(path))
        try:
            payload = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=self.device)
        if isinstance(payload, dict):
            state_dict = payload.get("model_state_dict", payload.get("state_dict", payload))
        else:
            state_dict = payload
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        print(
            "Loaded warmstart checkpoint {} | missing={} | unexpected={}".format(
                path,
                len(missing),
                len(unexpected),
            )
        )

    def checkpoint_path(self):
        name = "{}_best.pt".format(self.run_stem())
        return self.checkpoint_dir / name

    def save_checkpoint(self, best_epoch, best_valid_loss, optimizer_state_dict=None):
        if not self.args.save_checkpoint:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if optimizer_state_dict is None:
            optimizer_state_dict = self.optimizer.state_dict()
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer_state_dict,
            "args": serializable_args(self.args),
            "best_epoch": best_epoch,
            "best_valid_loss": safe_float(best_valid_loss),
        }
        path = self.checkpoint_path()
        torch.save(payload, path)
        print("Saved checkpoint: {}".format(path))

    def offset_target_day(self, offset):
        return offset + self.args.lookback_length + self.args.steps - 1

    def offset_regime_features(self, offset):
        seq_len = self.args.lookback_length
        prices = self.price_data[:, offset : offset + seq_len]
        return lookback_regime_feature_dict(prices)

    def fit_regime_stats(self, offsets):
        values = []
        for offset in offsets:
            features = self.offset_regime_features(int(offset))
            values.append([features[name] for name in REGIME_FEATURE_NAMES])
        matrix = np.asarray(values, dtype=float)
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
        std = np.where(std > 1e-8, std, 1.0)
        z = (matrix - mean[None, :]) / std[None, :]
        stress_k2 = z[:, 1] + z[:, 4] + z[:, 5] + z[:, 6]
        stress_k3 = z[:, 1] + z[:, 5]
        dispersion = z[:, 4]
        return {
            "feature_names": REGIME_FEATURE_NAMES,
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "stress_k2_q70": float(np.nanquantile(stress_k2, 0.70)),
            "stress_k3_q70": float(np.nanquantile(stress_k3, 0.70)),
            "dispersion_q70": float(np.nanquantile(dispersion, 0.70)),
        }

    def save_regime_thresholds(self):
        if not is_rcls_delta(self.args.model):
            return
        output = self.results_dir / "regime_thresholds_{}.json".format(self.run_stem())
        if self.regime_artifact is not None:
            payload = self.regime_artifact.to_metadata()
        else:
            payload = dict(self.regime_stats)
        payload.update(
            {
                "dataset": self.args.market,
                "model": self.args.model,
                "seed": self.result_seed,
                "num_regimes": self.args.num_regimes,
                "regime_mode": self.args.regime_mode,
                "label_source": "lookback_only_train_offsets",
            }
        )
        with output.open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def pseudo_regime_label(self, offset):
        if not is_rcls_delta(self.args.model) or self.args.num_regimes <= 1:
            return 0
        if self.args.regime_mode in NEW_REGIME_MODES:
            row = self.regime_row(offset)
            if row is None:
                return 0
            return int(row.get("regime_label", 0))
        features = self.offset_regime_features(offset)
        values = np.asarray([features[name] for name in REGIME_FEATURE_NAMES], dtype=float)
        mean = np.asarray(self.regime_stats["feature_mean"], dtype=float)
        std = np.asarray(self.regime_stats["feature_std"], dtype=float)
        z = (values - mean) / std
        if self.args.num_regimes == 2:
            stress = z[1] + z[4] + z[5] + z[6]
            return int(stress >= self.regime_stats["stress_k2_q70"])
        stress = z[1] + z[5]
        dispersion = z[4]
        if stress >= self.regime_stats["stress_k3_q70"]:
            return 1
        if dispersion >= self.regime_stats["dispersion_q70"]:
            return 2
        return 0

    def get_batch(self, offset):
        seq_len = self.args.lookback_length
        steps = self.args.steps
        mask_batch = self.mask_data[:, offset : offset + seq_len + steps]
        mask_batch = np.min(mask_batch, axis=1)
        return (
            self.eod_data[:, offset : offset + seq_len, :],
            np.expand_dims(mask_batch, axis=1),
            np.expand_dims(self.price_data[:, offset + seq_len - 1], axis=1),
            np.expand_dims(self.gt_data[:, offset + seq_len + steps - 1], axis=1),
        )

    def to_device(self, batch):
        return tuple(torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in batch)

    def lookback_stress_features(self, offset):
        features = self.offset_regime_features(offset)
        return (
            safe_float(features["market_ret_std"]),
            safe_float(features["synchronism"]),
            safe_float(features["dispersion"]),
            safe_float(features["mean_abs_ret"]),
        )

    def gate_values(self):
        pi = getattr(self.model, "last_regime_prob", None)
        if pi is None:
            return ["", "", ""]
        pi = pi.detach().cpu().numpy()
        if pi.ndim == 2:
            pi = pi[0]
        values = [safe_float(x) for x in pi.tolist()]
        while len(values) < 3:
            values.append("")
        return values[:3]

    def model_scalar(self, attr_name):
        value = getattr(self.model, attr_name, None)
        if value is None:
            return ""
        if torch.is_tensor(value):
            value = value.detach().cpu().item()
        return safe_float(value)

    def delta_scale_value(self):
        mixer = getattr(self.model, "stock_mixer", None)
        value = getattr(mixer, "delta_scale", None)
        if value is None:
            return ""
        if torch.is_tensor(value):
            value = value.detach().cpu().item()
        return safe_float(value)

    def gate_summary(self):
        values = self.gate_values()
        numeric = [float(x) for x in values if x != ""]
        entropy = gate_entropy_from_probs(numeric)
        if numeric:
            dominant = int(np.argmax(numeric))
        else:
            dominant = ""
        return values, entropy, dominant

    def gate_feature_row(self, split_name, cur_offset):
        values, entropy, dominant = self.gate_summary()
        features = self.offset_regime_features(cur_offset)
        regime_diag = self.regime_diagnostics(cur_offset)
        day_idx = self.offset_target_day(cur_offset)
        row = {
            "dataset": self.args.market,
            "model": self.args.model,
            "seed": self.result_seed,
            "day_idx": int(day_idx),
            "split": split_name,
            "regime_mode": regime_diag["regime_mode"],
            "regime_label": regime_diag["regime_label"],
            "regime_confidence": regime_diag["regime_confidence"],
            "regime_margin": regime_diag["regime_margin"],
            "breadth_score": regime_diag["breadth_score"],
            "stress_score": regime_diag["stress_score"],
            "corr_score": regime_diag["corr_score"],
            "regime_0": values[0],
            "regime_1": values[1],
            "regime_2": values[2],
            "gate_entropy": entropy,
            "dominant_regime": dominant,
            "delta_norm": self.model_scalar("last_delta_norm"),
            "base_norm": self.model_scalar("last_base_norm"),
            "delta_scale": self.delta_scale_value(),
            "pseudo_regime_label": (
                self.pseudo_regime_label(cur_offset) if is_rcls_delta(self.args.model) else ""
            ),
        }
        for name in REGIME_FEATURE_NAMES:
            row[name] = safe_float(features[name])
        return row

    def prediction_rows(self, split_name, cur_offset, result_offset, cur_rr, gt_batch, mask_batch):
        market_vol, synchronism, dispersion, mean_abs_ret = self.lookback_stress_features(
            cur_offset
        )
        gate_values, gate_entropy, dominant_regime = self.gate_summary()
        gate0, gate1, gate2 = gate_values
        day_idx = self.offset_target_day(cur_offset)
        regime_diag = self.regime_diagnostics(cur_offset)
        pred = cur_rr[:, 0].detach().cpu().numpy()
        target = gt_batch[:, 0].detach().cpu().numpy()
        mask = mask_batch[:, 0].detach().cpu().numpy()
        delta_norm = self.model_scalar("last_delta_norm")
        base_norm = self.model_scalar("last_base_norm")
        delta_scale = self.delta_scale_value()
        pseudo_label = (
            self.pseudo_regime_label(cur_offset) if is_rcls_delta(self.args.model) else ""
        )
        rows = []
        for stock_idx in range(self.config["stock_num"]):
            rows.append(
                {
                    "dataset": self.args.market,
                    "model": self.args.model,
                    "seed": self.result_seed,
                    "split": split_name,
                    "day_idx": int(day_idx),
                    "stock_idx": int(stock_idx),
                    "pred": safe_float(pred[stock_idx]),
                    "target": safe_float(target[stock_idx]),
                    "mask": safe_float(mask[stock_idx]),
                    "market_vol_lookback": market_vol,
                    "synchronism_lookback": synchronism,
                    "dispersion_lookback": dispersion,
                    "mean_abs_ret_lookback": mean_abs_ret,
                    "stress_source": "lookback",
                    "regime_mode": regime_diag["regime_mode"],
                    "regime_label": regime_diag["regime_label"],
                    "regime_confidence": regime_diag["regime_confidence"],
                    "regime_margin": regime_diag["regime_margin"],
                    "breadth_score": regime_diag["breadth_score"],
                    "stress_score": regime_diag["stress_score"],
                    "corr_score": regime_diag["corr_score"],
                    "regime_0": gate0,
                    "regime_1": gate1,
                    "regime_2": gate2,
                    "gate_entropy": gate_entropy,
                    "dominant_regime": dominant_regime,
                    "delta_norm": delta_norm,
                    "base_norm": base_norm,
                    "delta_scale": delta_scale,
                    "pseudo_regime_label": pseudo_label,
                }
            )
        return rows

    def evaluate_range(self, start_index, end_index, split_name=None, collect_rows=False):
        stock_num = self.config["stock_num"]
        eval_start = time.time()
        with torch.no_grad():
            cur_valid_pred = np.zeros([stock_num, end_index - start_index], dtype=float)
            cur_valid_gt = np.zeros([stock_num, end_index - start_index], dtype=float)
            cur_valid_mask = np.zeros([stock_num, end_index - start_index], dtype=float)
            rows = []
            gate_rows = []
            loss = 0.0
            reg_loss = 0.0
            rank_loss = 0.0
            offset_start = start_index - self.args.lookback_length - self.args.steps + 1
            offset_end = end_index - self.args.lookback_length - self.args.steps + 1
            for cur_offset in range(offset_start, offset_end):
                data_batch, mask_batch, price_batch, gt_batch = self.to_device(
                    self.get_batch(cur_offset)
                )
                prediction = self.model(data_batch, manual_pi=self.manual_pi_for_offset(cur_offset))
                cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = get_loss(
                    prediction,
                    gt_batch,
                    price_batch,
                    mask_batch,
                    stock_num,
                    self.args.alpha,
                )
                loss += cur_loss.item()
                reg_loss += cur_reg_loss.item()
                rank_loss += cur_rank_loss.item()
                result_offset = cur_offset - offset_start
                cur_valid_pred[:, result_offset] = cur_rr[:, 0].cpu()
                cur_valid_gt[:, result_offset] = gt_batch[:, 0].cpu()
                cur_valid_mask[:, result_offset] = mask_batch[:, 0].cpu()
                if collect_rows:
                    rows.extend(
                        self.prediction_rows(
                            split_name,
                            cur_offset,
                            result_offset,
                            cur_rr,
                            gt_batch,
                            mask_batch,
                        )
                    )
                    gate_rows.append(self.gate_feature_row(split_name, cur_offset))
            denom = end_index - start_index
            loss = loss / denom
            reg_loss = reg_loss / denom
            rank_loss = rank_loss / denom
            cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
        elapsed = time.time() - eval_start
        return loss, reg_loss, rank_loss, cur_valid_perf, rows, gate_rows, elapsed

    def validate(self, start_index, end_index):
        loss, reg_loss, rank_loss, perf, _, _, _ = self.evaluate_range(start_index, end_index)
        return loss, reg_loss, rank_loss, perf

    def save_best_predictions(self):
        valid_index = self.config["valid_index"]
        test_index = self.config["test_index"]
        self.model.eval()
        _, _, _, _, valid_rows, valid_gate_rows, _ = self.evaluate_range(
            valid_index, test_index, split_name="valid", collect_rows=True
        )
        _, _, _, _, test_rows, test_gate_rows, test_elapsed = self.evaluate_range(
            test_index, self.trade_dates, split_name="test", collect_rows=True
        )
        output = self.results_dir / "preds_{}.csv".format(self.run_stem())
        with output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=PREDICTION_COLUMNS)
            writer.writeheader()
            for row in valid_rows:
                writer.writerow(row)
            for row in test_rows:
                writer.writerow(row)
        gate_rows = valid_gate_rows + test_gate_rows
        if is_rcls_delta(self.args.model) and gate_rows:
            gate_output = self.results_dir / "gate_features_{}.csv".format(self.run_stem())
            with gate_output.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=GATE_FEATURE_COLUMNS)
                writer.writeheader()
                for row in gate_rows:
                    writer.writerow(row)
            print("Saved gate features: {}".format(gate_output))
        test_days = max(1, self.trade_dates - test_index)
        self.infer_time_ms_per_day = (test_elapsed / test_days) * 1000.0
        print("Saved predictions: {}".format(output))

    def append_metadata(
        self,
        epochs_ran,
        train_time_sec,
        total_time_sec,
        max_vram_gb,
        best_epoch,
        best_valid_loss,
    ):
        row = {
            "dataset": self.args.market,
            "model": self.args.model,
            "seed": self.result_seed,
            "epochs": self.args.epochs,
            "epochs_ran": epochs_ran,
            "patience": self.args.patience,
            "learning_rate": self.args.learning_rate,
            "batch_size": self.config["stock_num"],
            "num_params": self.num_params,
            "train_time_sec": safe_float(train_time_sec),
            "total_time_sec": safe_float(total_time_sec),
            "max_vram_gb": safe_float(max_vram_gb),
            "infer_time_ms_per_day": safe_float(self.infer_time_ms_per_day),
            "best_epoch": best_epoch,
            "best_valid_loss": safe_float(best_valid_loss),
            "numpy_seed": self.numpy_seed,
            "torch_seed": self.torch_seed,
            "git_commit": git_commit(),
            "device": self.device_name,
            "num_regimes": self.args.num_regimes,
            "regime_mode": self.args.regime_mode,
            "regime_temperature": safe_float(self.args.regime_temperature),
            "regime_jump_min_run": self.args.regime_jump_min_run,
            "regime_label_counts": (
                json.dumps(self.regime_artifact.label_counts, sort_keys=True)
                if self.regime_artifact is not None
                else ""
            ),
            "train_regime_switch_count": self.regime_summary_stats.get("train_regime_switch_count", ""),
            "valid_regime_switch_count": self.regime_summary_stats.get("valid_regime_switch_count", ""),
            "test_regime_switch_count": self.regime_summary_stats.get("test_regime_switch_count", ""),
            "train_regime_occupancy_min": self.regime_summary_stats.get("train_regime_occupancy_min", ""),
            "valid_regime_occupancy_min": self.regime_summary_stats.get("valid_regime_occupancy_min", ""),
            "test_regime_occupancy_min": self.regime_summary_stats.get("test_regime_occupancy_min", ""),
            "delta_scale": safe_float(self.args.delta_scale),
            "delta_trainable_scale": self.args.delta_trainable_scale,
            "delta_dropout": safe_float(self.args.delta_dropout),
            "gate_temperature": safe_float(self.args.gate_temperature),
            "gate_feature_mode": self.args.gate_feature_mode,
            "uniform_gate": self.args.uniform_gate,
            "gate_pseudo_label": self.args.gate_pseudo_label,
            "gate_pseudo_weight": safe_float(self.args.gate_pseudo_weight),
            "gate_pseudo_final_weight": safe_float(self.args.gate_pseudo_final_weight),
            "gate_pseudo_warmup_epochs": self.args.gate_pseudo_warmup_epochs,
            "gate_confidence_weight": safe_float(self.args.gate_confidence_weight),
            "expert_diversity_weight": safe_float(self.args.expert_diversity_weight),
            "freeze_base_epochs": self.args.freeze_base_epochs,
            "warmstart_checkpoint": self.args.warmstart_checkpoint,
            "save_checkpoint": self.args.save_checkpoint,
            "checkpoint_dir": str(self.checkpoint_dir),
        }
        upsert_csv_row(
            self.results_dir / "run_metadata.csv",
            METADATA_COLUMNS,
            row,
            ["dataset", "model", "seed", "regime_mode"],
        )

    def set_base_freeze(self, epoch):
        mixer = getattr(self.model, "stock_mixer", None)
        base = getattr(mixer, "base_stock_mixer", None)
        if base is None:
            return
        freeze = epoch < self.args.freeze_base_epochs
        for param in base.parameters():
            param.requires_grad = not freeze

    def auxiliary_loss(self, offset, epoch):
        if not is_rcls_delta(self.args.model):
            return torch.tensor(0.0, device=self.device), {"gate": 0.0, "conf": 0.0, "div": 0.0}

        aux_loss = torch.tensor(0.0, device=self.device)
        components = {"gate": 0.0, "conf": 0.0, "div": 0.0}
        logits = getattr(self.model, "last_gate_logits", None)
        if self.args.gate_pseudo_label and logits is not None and logits.numel() > 1:
            label = self.pseudo_regime_label(offset)
            label = min(label, logits.numel() - 1)
            label_t = torch.tensor([label], dtype=torch.long, device=logits.device)
            logits_t = logits.view(1, -1)
            if epoch < self.args.gate_pseudo_warmup_epochs:
                weight = self.args.gate_pseudo_weight
            else:
                weight = self.args.gate_pseudo_final_weight
            gate_loss = F.cross_entropy(logits_t, label_t)
            aux_loss = aux_loss + weight * gate_loss
            components["gate"] = float((weight * gate_loss).detach().cpu())

        pi = getattr(self.model, "last_regime_prob", None)
        if self.args.gate_confidence_weight > 0.0 and pi is not None and pi.numel() > 1:
            probs = torch.clamp(pi, 1e-12, 1.0)
            entropy = -(probs * probs.log()).sum()
            conf_loss = self.args.gate_confidence_weight * entropy
            aux_loss = aux_loss + conf_loss
            components["conf"] = float(conf_loss.detach().cpu())

        mixer = getattr(self.model, "stock_mixer", None)
        if self.args.expert_diversity_weight > 0.0 and hasattr(mixer, "expert_diversity_loss"):
            div = mixer.expert_diversity_loss()
            div_loss = self.args.expert_diversity_weight * div
            aux_loss = aux_loss + div_loss
            components["div"] = float(div_loss.detach().cpu())

        return aux_loss, components

    def write_epoch_diagnostics(self, rows):
        if not rows:
            return
        output = self.results_dir / "epoch_diagnostics_{}.csv".format(self.run_stem())
        with output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EPOCH_DIAGNOSTIC_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def train(self):
        stock_num = self.config["stock_num"]
        valid_index = self.config["valid_index"]
        test_index = self.config["test_index"]
        train_steps = len(self.train_offsets)
        best_valid_loss = np.inf
        best_epoch = None
        best_valid_perf = None
        best_test_perf = None
        best_state = None
        best_optimizer_state = None
        epochs_without_improvement = 0
        train_time_sec = 0.0
        epochs_ran = 0
        epoch_diagnostics = []

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        total_start = time.time()

        for epoch in range(self.args.epochs):
            epochs_ran = epoch + 1
            print("epoch{}##########################################################".format(epoch + 1))
            epoch_offsets = self.train_offsets.copy()
            np.random.shuffle(epoch_offsets)
            self.set_base_freeze(epoch)
            self.model.train()
            epoch_train_start = time.time()
            tra_total_loss = 0.0
            tra_main_loss = 0.0
            tra_reg_loss = 0.0
            tra_rank_loss = 0.0
            tra_aux_loss = 0.0
            tra_gate_loss = 0.0
            tra_conf_loss = 0.0
            tra_div_loss = 0.0
            gate_entropies = []
            gate_probs = []
            pseudo_counts = {}
            for j in range(train_steps):
                offset = int(epoch_offsets[j])
                target_day = self.offset_target_day(offset)
                if target_day >= valid_index:
                    raise RuntimeError(
                        "Training offset {} targets validation day {}".format(
                            offset,
                            target_day,
                        )
                    )
                data_batch, mask_batch, price_batch, gt_batch = self.to_device(
                    self.get_batch(offset)
                )
                self.optimizer.zero_grad()
                prediction = self.model(data_batch, manual_pi=self.manual_pi_for_offset(offset))
                cur_loss, cur_reg_loss, cur_rank_loss, _ = get_loss(
                    prediction,
                    gt_batch,
                    price_batch,
                    mask_batch,
                    stock_num,
                    self.args.alpha,
                )
                aux_loss, aux_components = self.auxiliary_loss(offset, epoch)
                total_loss = cur_loss + aux_loss
                total_loss.backward()
                self.optimizer.step()

                if is_rcls_delta(self.args.model):
                    label = self.pseudo_regime_label(offset)
                    pseudo_counts[label] = pseudo_counts.get(label, 0) + 1
                pi = getattr(self.model, "last_regime_prob", None)
                if pi is not None:
                    pi_np = pi.detach().cpu().numpy()
                    if pi_np.ndim == 2:
                        pi_np = pi_np[0]
                    pi_np = np.asarray(pi_np, dtype=float)
                    gate_probs.append(pi_np)
                    entropy = gate_entropy_from_probs(pi_np.tolist())
                    if entropy != "":
                        gate_entropies.append(float(entropy))

                tra_total_loss += total_loss.item()
                tra_main_loss += cur_loss.item()
                tra_reg_loss += cur_reg_loss.item()
                tra_rank_loss += cur_rank_loss.item()
                tra_aux_loss += aux_loss.item()
                tra_gate_loss += aux_components["gate"]
                tra_conf_loss += aux_components["conf"]
                tra_div_loss += aux_components["div"]
            train_time_sec += time.time() - epoch_train_start
            tra_total_loss = tra_total_loss / train_steps
            tra_main_loss = tra_main_loss / train_steps
            tra_reg_loss = tra_reg_loss / train_steps
            tra_rank_loss = tra_rank_loss / train_steps
            tra_aux_loss = tra_aux_loss / train_steps
            tra_gate_loss = tra_gate_loss / train_steps
            tra_conf_loss = tra_conf_loss / train_steps
            tra_div_loss = tra_div_loss / train_steps
            print(
                "Train : loss:{:.2e}  =  {:.2e} + alpha*{:.2e} + aux:{:.2e}".format(
                    tra_total_loss, tra_reg_loss, tra_rank_loss, tra_aux_loss
                )
            )

            mean_pi = ["", "", ""]
            if gate_probs:
                max_width = max(len(values) for values in gate_probs)
                padded = np.full((len(gate_probs), max_width), np.nan, dtype=float)
                for idx, values in enumerate(gate_probs):
                    padded[idx, : len(values)] = values
                means = np.nanmean(padded, axis=0).tolist()
                for idx, value in enumerate(means[:3]):
                    mean_pi[idx] = safe_float(value)
            if is_rcls_delta(self.args.model):
                diagnostic = {
                    "dataset": self.args.market,
                    "model": self.args.model,
                    "seed": self.result_seed,
                    "epoch": epoch + 1,
                    "regime_mode": self.args.regime_mode,
                    "loss_main": safe_float(tra_main_loss),
                    "loss_gate_ce": safe_float(tra_gate_loss),
                    "loss_gate_confidence": safe_float(tra_conf_loss),
                    "loss_expert_diversity": safe_float(tra_div_loss),
                    "loss_total": safe_float(tra_total_loss),
                    "mean_gate_entropy": (
                        safe_float(np.mean(gate_entropies)) if gate_entropies else ""
                    ),
                    "std_gate_entropy": (
                        safe_float(np.std(gate_entropies)) if gate_entropies else ""
                    ),
                    "mean_pi_0": mean_pi[0],
                    "mean_pi_1": mean_pi[1],
                    "mean_pi_2": mean_pi[2],
                    "pseudo_label_counts": json.dumps(pseudo_counts, sort_keys=True),
                }
                epoch_diagnostics.append(diagnostic)
                self.write_epoch_diagnostics(epoch_diagnostics)

            self.model.eval()
            val_loss, val_reg_loss, val_rank_loss, val_perf = self.validate(
                valid_index, test_index
            )
            print(
                "Valid : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}".format(
                    val_loss, val_reg_loss, val_rank_loss
                )
            )

            test_loss, test_reg_loss, test_rank_loss, test_perf = self.validate(
                test_index, self.trade_dates
            )
            print(
                "Test: loss:{:.2e}  =  {:.2e} + alpha*{:.2e}".format(
                    test_loss, test_reg_loss, test_rank_loss
                )
            )

            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                best_epoch = epoch + 1
                best_valid_perf = val_perf
                best_test_perf = test_perf
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                best_optimizer_state = self.optimizer.state_dict()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print("Valid performance:\n", format_metrics(val_perf))
            print("Test performance:\n", format_metrics(test_perf), "\n")

            if self.args.patience and epochs_without_improvement >= self.args.patience:
                print(
                    "Early stopping after {} epochs without validation improvement.".format(
                        self.args.patience
                    )
                )
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        if self.args.save_predictions:
            self.save_best_predictions()

        total_time_sec = time.time() - total_start
        if self.device.type == "cuda":
            max_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
        else:
            max_vram_gb = 0.0
        self.append_metadata(
            epochs_ran,
            train_time_sec,
            total_time_sec,
            max_vram_gb,
            best_epoch,
            best_valid_loss,
        )
        self.save_checkpoint(best_epoch, best_valid_loss, best_optimizer_state)

        return {
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid_loss,
            "best_valid_perf": best_valid_perf,
            "best_test_perf": best_test_perf,
            "epochs_ran": epochs_ran,
        }


def run_once(args, run_index, device, data, device_name):
    numpy_seed = args.numpy_seed + run_index
    torch_seed = args.torch_seed + run_index
    set_seeds(numpy_seed, torch_seed)

    config = MARKET_CONFIGS[args.market]
    trainer = Trainer(args, config, device, data, run_index, device_name)
    print(
        (
            "Run {}/{} | market={} | model={} | activation={} | main={} | scale={} | "
            "stock={} | epochs={} | patience={} | numpy_seed={} | torch_seed={}"
        ).format(
            run_index + 1,
            args.runs,
            args.market,
            args.model,
            args.activation,
            args.main_mixer_activation,
            args.scale_mixer_activation,
            args.stock_activation,
            args.epochs,
            args.patience,
            numpy_seed,
            torch_seed,
        )
    )
    print("Trainable parameters: {}".format(trainer.num_params))
    return trainer.train()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(parents=True, exist_ok=True)
    if args.save_checkpoint:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(json.dumps(selected_config(args), indent=2, sort_keys=True))
        return
    print_selected_config(args)
    device, device_name = get_device(args.require_gpu)
    print("Using device: {} ({})".format(device, device_name))
    print("Dataset root: {}".format(args.dataset_root.resolve()))
    print("Output root: {}".format(args.output_root.resolve()))
    print("Results dir: {}".format(args.results_dir.resolve()))

    data = load_market_data(args.dataset_root, args.market, args.steps)
    results = []
    for run_index in range(args.runs):
        result = run_once(args, run_index, device, data, device_name)
        results.append(result)
        print("Best epoch: {}".format(result["best_epoch"]))
        print("Best validation loss: {:.2e}".format(result["best_valid_loss"]))
        print("Best validation performance:\n", format_metrics(result["best_valid_perf"]))
        print("Best test performance:\n", format_metrics(result["best_test_perf"]))
        print_paper_comparison(args.market, result["best_test_perf"])

    if args.runs > 1:
        metric_names = ["mse", "IC", "RIC", "prec_10", "sharpe5"]
        print("Average best test performance across {} runs:".format(args.runs))
        averages = {
            name: np.mean([result["best_test_perf"][name] for result in results])
            for name in metric_names
        }
        print(format_metrics(averages))
        print_paper_comparison(args.market, averages)


if __name__ == "__main__":
    main()
