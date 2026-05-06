from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model_rcls_cag import RCLSForecastModel, get_price_loss, parameter_count
from rcls_context import CONTEXT_FEATURES, FIVE_CONTEXT_FEATURES, build_context_table
from rcls_metrics import aggregate_daily_metrics


MARKET_CONFIGS = {
    "NASDAQ": {"stock_num": 1026, "valid_index": 756, "test_index": 1008, "market_num": 20},
    "SP500": {"stock_num": 474, "valid_index": 1006, "test_index": 1259, "market_num": 8},
    "S&P500": {"alias": "SP500"},
}

MODEL_NAMES = [
    "stockmixer",
    "cag_mlp",
    "rcls_cag_k1",
    "rcls_cag_k2",
    "rcls_cag_k2_uniform",
    "rcls_cag_k2_nocontext",
    "rcls_cag_k2_nogate",
    "rcls_cag_k3",
    "rcls_cag_k3_uniform",
]


def parse_args(argv=None):
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="RCLS-CAG NeurIPS experiment runner.")
    parser.add_argument("--dataset", default="NASDAQ", choices=sorted(MARKET_CONFIGS))
    parser.add_argument("--dataset-root", type=Path, default=repo_root / "dataset")
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--numpy-seed", type=int, default=123456789)
    parser.add_argument("--torch-seed", type=int, default=12345678)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--alpha-rank", type=float, default=0.1)
    parser.add_argument("--lookback-length", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--market-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--router-hidden", type=int, default=64)
    parser.add_argument("--router-temperature", type=float, default=0.7)
    parser.add_argument("--context-normalization", choices=["minmax", "zscore"], default="minmax")
    parser.add_argument("--context-mode", choices=["five", "full", "none"], default="five")
    parser.add_argument("--router-ce-weight", type=float, default=0.02)
    parser.add_argument("--router-balance-weight", type=float, default=0.001)
    parser.add_argument("--router-confidence-weight", type=float, default=0.0005)
    parser.add_argument("--expert-diversity-weight", type=float, default=0.00001)
    parser.add_argument("--activation", default="hardswish", choices=["hardswish", "relu", "gelu"])
    parser.add_argument("--main-mixer-activation", default="hardswish", choices=["hardswish", "relu", "gelu"])
    parser.add_argument("--scale-mixer-activation", default="gelu", choices=["hardswish", "relu", "gelu"])
    parser.add_argument("--stock-activation", default="hardswish", choices=["hardswish", "relu", "gelu"])
    parser.add_argument("--require-gpu", default="3090")
    parser.add_argument("--output-root", type=Path, default=repo_root / "Exp" / "RCLS" / "results" / "paper15_seed1_match")
    parser.add_argument("--checkpoint-root", type=Path, default=repo_root / "Exp" / "RCLS" / "checkpoints" / "paper15_seed1_match")
    return parser.parse_args(argv)


def canonical_dataset(name):
    cfg = MARKET_CONFIGS[name]
    if "alias" in cfg:
        return cfg["alias"]
    return name


def get_config(dataset):
    return MARKET_CONFIGS[canonical_dataset(dataset)]


def get_device(required):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    torch.cuda.set_device(0)
    name = torch.cuda.get_device_name(0)
    if required and str(required).lower() not in name.lower():
        raise RuntimeError(f"cuda:0 is '{name}', expected substring '{required}'.")
    return torch.device("cuda:0"), name


def set_seeds(np_seed, torch_seed):
    random.seed(np_seed)
    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_market_data(dataset_root, dataset, steps):
    dataset = canonical_dataset(dataset)
    if dataset == "SP500":
        data = np.load(dataset_root / "SP500" / "SP500.npy")
        data = data[:, 915:, :]
        price_data = data[:, :, -1]
        mask_data = np.ones((data.shape[0], data.shape[1]), dtype=np.float32)
        gt_data = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
        for ticket in range(data.shape[0]):
            for row in range(1, data.shape[1]):
                gt_data[ticket][row] = (data[ticket][row][-1] - data[ticket][row - steps][-1]) / data[ticket][row - steps][-1]
        return data.astype(np.float32), mask_data, gt_data, price_data.astype(np.float32)

    path = dataset_root / dataset
    with (path / "eod_data.pkl").open("rb") as f:
        eod = pickle.load(f)
    with (path / "mask_data.pkl").open("rb") as f:
        mask = pickle.load(f)
    with (path / "gt_data.pkl").open("rb") as f:
        gt = pickle.load(f)
    with (path / "price_data.pkl").open("rb") as f:
        price = pickle.load(f)
    return np.asarray(eod, dtype=np.float32), np.asarray(mask, dtype=np.float32), np.asarray(gt, dtype=np.float32), np.asarray(price, dtype=np.float32)


def build_offsets(valid_index, test_index, total_days, lookback, steps):
    return {
        "train": np.arange(0, valid_index - lookback - steps + 1, dtype=int),
        "valid": np.arange(valid_index - lookback, test_index - lookback, dtype=int),
        "test": np.arange(test_index - lookback, total_days - lookback - steps + 1, dtype=int),
    }


def context_columns_for_mode(mode):
    if mode == "none":
        return []
    names = FIVE_CONTEXT_FEATURES if mode == "five" else CONTEXT_FEATURES
    return [f"ctx_{name}" for name in names]


class Trainer:
    def __init__(self, args, config, device, device_name, data):
        self.args = args
        self.config = config
        self.device = device
        self.device_name = device_name
        self.eod_data, self.mask_data, self.gt_data, self.price_data = data
        self.total_days = self.mask_data.shape[1]
        self.offsets = build_offsets(config["valid_index"], config["test_index"], self.total_days, args.lookback_length, args.steps)
        self.context_df, self.raw_context_df, self.normalizer, self.regime_stats = build_context_table(
            self.eod_data,
            self.mask_data,
            self.offsets,
            args.lookback_length,
            args.steps,
            normalization=args.context_normalization,
        )
        self.context_lookup = {int(row["offset"]): row.to_dict() for _, row in self.context_df.iterrows()}
        self.context_cols = context_columns_for_mode(args.context_mode)
        self.market_dim = args.market_dim or config["market_num"]
        self.model = RCLSForecastModel(
            model_name=args.model,
            stocks=config["stock_num"],
            time_steps=args.lookback_length,
            channels=self.eod_data.shape[2],
            market_dim=self.market_dim,
            scale=3,
            context_dim=max(len(self.context_cols), 1),
            activation=args.activation,
            main_mixer_activation=args.main_mixer_activation,
            scale_mixer_activation=args.scale_mixer_activation,
            stock_activation=args.stock_activation,
            router_hidden=args.router_hidden,
            router_temperature=args.router_temperature,
            num_layers=args.num_layers,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def context_tensor(self, offset):
        if not self.context_cols:
            values = np.zeros(1, dtype=np.float32)
        else:
            row = self.context_lookup[int(offset)]
            values = np.asarray([row[col] for col in self.context_cols], dtype=np.float32)
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def batch(self, offset):
        seq = self.args.lookback_length
        steps = self.args.steps
        mask = np.min(self.mask_data[:, offset : offset + seq + steps], axis=1)
        return (
            torch.as_tensor(self.eod_data[:, offset : offset + seq, :], dtype=torch.float32, device=self.device),
            torch.as_tensor(mask[:, None], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.price_data[:, offset + seq - 1, None], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.gt_data[:, offset + seq + steps - 1, None], dtype=torch.float32, device=self.device),
            self.context_tensor(offset),
        )

    def aux_loss(self, aux, offset):
        pi = aux.get("pi")
        if pi is None or pi.numel() <= 1:
            return next(self.model.parameters()).new_tensor(0.0), {}
        row = self.context_lookup[int(offset)]
        k = int(pi.numel())
        label_key = "regime_label_k3" if k == 3 else "regime_label_k2"
        target = torch.tensor([min(int(row[label_key]), k - 1)], dtype=torch.long, device=self.device)
        ce = F.nll_loss(torch.log(pi.view(1, -1) + 1e-12), target)
        balance = k * torch.sum(pi.pow(2))
        confidence = aux["gate_entropy"]
        diversity = self.model.auxiliary_diversity_loss()
        loss = (
            self.args.router_ce_weight * ce
            + self.args.router_balance_weight * balance
            + self.args.router_confidence_weight * confidence
            + self.args.expert_diversity_weight * diversity
        )
        return loss, {
            "router_ce": float(ce.detach().cpu()),
            "router_balance": float(balance.detach().cpu()),
            "router_confidence": float(confidence.detach().cpu()),
            "expert_diversity": float(diversity.detach().cpu()),
        }

    def run_offsets(self, split, offsets, collect=False):
        self.model.eval()
        total_loss = 0.0
        total_point = 0.0
        total_rank = 0.0
        records = []
        with torch.no_grad():
            for offset in offsets:
                x, mask, base, target, ctx = self.batch(int(offset))
                pred_price, aux = self.model(x, ctx)
                loss, point, rank, pred_return = get_price_loss(pred_price, target, base, mask, self.config["stock_num"], self.args.alpha_rank)
                total_loss += float(loss.detach().cpu())
                total_point += float(point.detach().cpu())
                total_rank += float(rank.detach().cpu())
                if collect:
                    records.extend(self.prediction_records(split, int(offset), pred_return, target, mask, aux))
        denom = max(len(offsets), 1)
        if collect:
            df = pd.DataFrame.from_records(records)
            metrics, _ = aggregate_daily_metrics(df)
        else:
            df = pd.DataFrame()
            metrics = {}
        return {
            "loss": total_loss / denom,
            "point_loss": total_point / denom,
            "rank_loss": total_rank / denom,
            "metrics": metrics,
            "predictions": df,
        }

    def prediction_records(self, split, offset, pred_return, target, mask, aux):
        pred_np = pred_return[:, 0].detach().cpu().numpy()
        target_np = target[:, 0].detach().cpu().numpy()
        mask_np = mask[:, 0].detach().cpu().numpy()
        row = self.context_lookup[offset]
        pi = aux.get("pi", torch.ones(1, device=self.device)).detach().cpu().numpy().astype(float)
        pi_fixed = np.full(3, np.nan, dtype=float)
        pi_fixed[: min(3, len(pi))] = pi[:3]
        entropy = float(aux.get("gate_entropy", torch.tensor(0.0)).detach().cpu())
        dominant = int(np.nanargmax(np.nan_to_num(pi_fixed, nan=-1.0)))
        day_idx = int(offset + self.args.lookback_length + self.args.steps - 1)
        ctx_values = {col: row[col] for col in row if col.startswith("ctx_")}
        out = []
        confidence = -entropy if len(pi) > 1 else np.nan
        for stock_idx in range(len(pred_np)):
            item = {
                "dataset": canonical_dataset(self.args.dataset),
                "model": self.args.model,
                "seed": self.args.seed,
                "split": split,
                "day_idx": day_idx,
                "offset": offset,
                "stock_idx": stock_idx,
                "pred": pred_np[stock_idx],
                "target": target_np[stock_idx],
                "return_ratio": pred_np[stock_idx],
                "mask": mask_np[stock_idx],
                "mu": pred_np[stock_idx],
                "sigma": np.nan,
                "confidence": confidence if np.isfinite(confidence) else abs(pred_np[stock_idx]),
                "regime_0": pi_fixed[0],
                "regime_1": pi_fixed[1],
                "regime_2": pi_fixed[2],
                "gate_entropy": entropy,
                "dominant_regime": dominant,
                "pseudo_regime_label_k2": int(row["regime_label_k2"]),
                "pseudo_regime_label_k3": int(row["regime_label_k3"]),
                "stress_score": row["stress_score"],
            }
            item.update(ctx_values)
            out.append(item)
        return out

    def train(self):
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        best_loss = float("inf")
        best_epoch = 0
        best_state = None
        epoch_rows = []
        patience_left = self.args.patience
        train_offsets = self.offsets["train"].copy()
        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            np.random.shuffle(train_offsets)
            loss_sum = 0.0
            point_sum = 0.0
            rank_sum = 0.0
            aux_sum = 0.0
            for offset in train_offsets:
                x, mask, base, target, ctx = self.batch(int(offset))
                self.optimizer.zero_grad()
                pred, aux = self.model(x, ctx)
                base_loss, point, rank, _ = get_price_loss(pred, target, base, mask, self.config["stock_num"], self.args.alpha_rank)
                aux_loss, _ = self.aux_loss(aux, int(offset))
                loss = base_loss + aux_loss
                loss.backward()
                self.optimizer.step()
                loss_sum += float(loss.detach().cpu())
                point_sum += float(point.detach().cpu())
                rank_sum += float(rank.detach().cpu())
                aux_sum += float(aux_loss.detach().cpu())
            valid = self.run_offsets("valid", self.offsets["valid"], collect=False)
            test = self.run_offsets("test", self.offsets["test"], collect=True)
            valid_eval = self.run_offsets("valid", self.offsets["valid"], collect=True)
            row = {
                "epoch": epoch,
                "train_loss": loss_sum / len(train_offsets),
                "train_point_loss": point_sum / len(train_offsets),
                "train_rank_loss": rank_sum / len(train_offsets),
                "train_aux_loss": aux_sum / len(train_offsets),
                "valid_loss": valid["loss"],
                "test_loss": test["loss"],
            }
            for prefix, values in (("valid", valid_eval["metrics"]), ("test", test["metrics"])):
                for key, value in values.items():
                    row[f"{prefix}_{key}"] = value
            epoch_rows.append(row)
            print(
                f"epoch {epoch:03d} | train={row['train_loss']:.4e} "
                f"valid={valid['loss']:.4e} test_ic={row.get('test_ic', np.nan):.5f} "
                f"test_rankic={row.get('test_rankic', np.nan):.5f} test_p10={row.get('test_p10', np.nan):.5f}",
                flush=True,
            )
            if valid["loss"] < best_loss:
                best_loss = valid["loss"]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience_left = self.args.patience
            elif self.args.patience > 0:
                patience_left -= 1
                if patience_left <= 0:
                    break
        if best_state is None:
            raise RuntimeError("No best model state was saved.")
        self.model.load_state_dict(best_state)
        valid_best = self.run_offsets("valid", self.offsets["valid"], collect=True)
        test_best = self.run_offsets("test", self.offsets["test"], collect=True)
        train_time = time.time() - start_time
        max_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
        return {
            "best_epoch": best_epoch,
            "best_valid_loss": valid_best["loss"],
            "best_test_loss": test_best["loss"],
            "valid_predictions": valid_best["predictions"],
            "test_predictions": test_best["predictions"],
            "valid_metrics": valid_best["metrics"],
            "test_metrics": test_best["metrics"],
            "epoch_rows": epoch_rows,
            "state_dict": best_state,
            "train_time_sec": train_time,
            "total_time_sec": train_time,
            "max_vram_gb": max_vram,
        }


def append_metadata(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(val) for val in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def write_outputs(args, trainer, result):
    out = args.output_root
    ckpt = args.checkpoint_root
    out.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}_{canonical_dataset(args.dataset)}_seed{args.seed}"
    preds = pd.concat([result["valid_predictions"], result["test_predictions"]], ignore_index=True)
    preds.to_csv(out / f"preds_{stem}.csv", index=False)
    pd.DataFrame(result["epoch_rows"]).to_csv(out / f"epoch_diagnostics_{stem}.csv", index=False)
    trainer.context_df.to_csv(out / f"context_table_{canonical_dataset(args.dataset)}.csv", index=False)
    trainer.raw_context_df.to_csv(out / f"context_table_raw_{canonical_dataset(args.dataset)}.csv", index=False)
    trainer.normalizer.save_json(out / f"context_normalizer_{canonical_dataset(args.dataset)}.json")
    torch.save({"model_state_dict": result["state_dict"], "args": vars(args), "best_epoch": result["best_epoch"]}, ckpt / f"best_model_{stem}.pt")
    torch.save({"model_state_dict": trainer.model.state_dict(), "args": vars(args)}, ckpt / f"last_model_{stem}.pt")
    row = {
        "baseline_source": "baseline_stockmixer_repro_seed1.md",
        "dataset": canonical_dataset(args.dataset),
        "model": args.model,
        "seed": args.seed,
        "numpy_seed": args.numpy_seed,
        "torch_seed": args.torch_seed,
        "epochs": args.epochs,
        "best_epoch": result["best_epoch"],
        "activation": args.activation,
        "main_mixer_activation": args.main_mixer_activation,
        "scale_mixer_activation": args.scale_mixer_activation,
        "stock_activation": args.stock_activation,
        "lookback_length": args.lookback_length,
        "steps": args.steps,
        "lr": args.lr,
        "alpha_rank": args.alpha_rank,
        "market_dim": trainer.market_dim,
        "num_layers": args.num_layers,
        "context_mode": args.context_mode,
        "context_normalization": args.context_normalization,
        "context_fit_split": "train",
        "router_hidden": args.router_hidden,
        "router_temperature": args.router_temperature,
        "router_ce_weight": args.router_ce_weight,
        "router_balance_weight": args.router_balance_weight,
        "router_confidence_weight": args.router_confidence_weight,
        "expert_diversity_weight": args.expert_diversity_weight,
        "train_offset_start": int(trainer.offsets["train"][0]),
        "train_offset_end": int(trainer.offsets["train"][-1]),
        "valid_offset_start": int(trainer.offsets["valid"][0]),
        "valid_offset_end": int(trainer.offsets["valid"][-1]),
        "test_offset_start": int(trainer.offsets["test"][0]),
        "test_offset_end": int(trainer.offsets["test"][-1]),
        "valid_index": trainer.config["valid_index"],
        "test_index": trainer.config["test_index"],
        "num_params": parameter_count(trainer.model),
        "train_time_sec": result["train_time_sec"],
        "total_time_sec": result["total_time_sec"],
        "max_vram_gb": result["max_vram_gb"],
        "best_valid_loss": result["best_valid_loss"],
        "best_test_loss": result["best_test_loss"],
    }
    for prefix, metrics in (("best_valid", result["valid_metrics"]), ("best_test", result["test_metrics"])):
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value
    append_metadata(out / "run_metadata.csv", row)
    with (out / f"config_{stem}.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe({"args": vars(args), "regime_stats": trainer.regime_stats}), f, indent=2)


def main(argv=None):
    args = parse_args(argv)
    args.dataset_root = args.dataset_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    device, device_name = get_device(args.require_gpu)
    set_seeds(args.numpy_seed, args.torch_seed)
    data = load_market_data(args.dataset_root, args.dataset, args.steps)
    trainer = Trainer(args, get_config(args.dataset), device, device_name, data)
    print(f"device={device_name} model={args.model} dataset={canonical_dataset(args.dataset)} params={parameter_count(trainer.model)}", flush=True)
    result = trainer.train()
    write_outputs(args, trainer, result)
    print(
        f"best_epoch={result['best_epoch']} test_ic={result['test_metrics'].get('ic', np.nan):.6f} "
        f"test_rankic={result['test_metrics'].get('rankic', np.nan):.6f} test_p10={result['test_metrics'].get('p10', np.nan):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
