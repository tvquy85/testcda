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

from context import FEATURE_NAMES, build_context_bundle, save_context_outputs
from model_cag import CAGStockMixer, StockMixer, get_loss, parameter_count


MARKET_CONFIGS = {
    "NASDAQ": {"stock_num": 1026, "valid_index": 756, "test_index": 1008, "market_num": 20},
    "SP500": {"stock_num": 474, "valid_index": 1006, "test_index": 1259, "market_num": 8},
}

PAPER_REFERENCE = {
    "NASDAQ": {
        "paper_stockmixer_IC": 0.041,
        "paper_stockmixer_RIC": 0.507,
        "paper_stockmixer_prec_10": 0.531,
        "paper_stockmixer_sharpe5": 1.418,
        "paper_cag_IC": 0.052,
        "paper_cag_RIC": 0.704,
        "paper_cag_prec_10": 0.541,
        "paper_cag_sharpe5": 1.407,
    }
}


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Clean matched StockMixer vs CAG-MLP runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=["stockmixer", "cag_gated_context"], required=True)
    parser.add_argument("--market", choices=sorted(MARKET_CONFIGS), default="NASDAQ")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--dataset-root", type=Path, default=repo_root / "dataset")
    parser.add_argument("--results-dir", type=Path, default=repo_root / "Exp" / "results" / "cag_context_matched60")
    parser.add_argument("--logs-dir", type=Path, default=repo_root / "Exp" / "logs" / "cag_context_matched60")
    parser.add_argument("--require-gpu", default="RTX 3090")
    parser.add_argument("--lookback-length", type=int, default=16)
    parser.add_argument("--fea-num", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--scale-factor", type=int, default=3)
    parser.add_argument("--activation", default="hardswish", choices=["hardswish", "relu", "gelu"])
    parser.add_argument("--main-mixer-activation", default=None, choices=["hardswish", "relu", "gelu"])
    parser.add_argument("--scale-mixer-activation", default=None, choices=["hardswish", "relu", "gelu"])
    parser.add_argument("--stock-activation", default="hardswish", choices=["hardswish", "relu", "gelu"])
    parser.add_argument("--cag-depth", type=int, default=3)
    parser.add_argument("--cag-dropout", type=float, default=0.0)
    parser.add_argument("--numpy-seed", type=int, default=123456789)
    parser.add_argument("--torch-seed", type=int, default=12345678)
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.main_mixer_activation is None:
        args.main_mixer_activation = args.activation
    if args.scale_mixer_activation is None:
        args.scale_mixer_activation = args.activation
    return args


def get_device(required_gpu):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    torch.cuda.set_device(0)
    name = torch.cuda.get_device_name(0)
    if required_gpu and required_gpu not in name:
        raise RuntimeError(f"cuda:0 is '{name}', expected substring '{required_gpu}'.")
    return torch.device("cuda:0"), name


def set_seeds(numpy_seed, torch_seed):
    random.seed(numpy_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_market_data(dataset_root: Path, market_name: str, steps: int):
    dataset_root = dataset_root.resolve()
    if market_name == "SP500":
        data = np.load(dataset_root / "SP500" / "SP500.npy")
        data = data[:, 915:, :]
        price_data = data[:, :, -1]
        mask_data = np.ones((data.shape[0], data.shape[1]), dtype=np.float32)
        eod_data = data
        gt_data = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
        for ticket in range(data.shape[0]):
            for row in range(1, data.shape[1]):
                gt_data[ticket][row] = (data[ticket][row][-1] - data[ticket][row - steps][-1]) / data[ticket][row - steps][-1]
        return eod_data, mask_data, gt_data, price_data

    dataset_path = dataset_root / market_name
    with (dataset_path / "eod_data.pkl").open("rb") as f:
        eod_data = pickle.load(f)
    with (dataset_path / "mask_data.pkl").open("rb") as f:
        mask_data = pickle.load(f)
    with (dataset_path / "gt_data.pkl").open("rb") as f:
        gt_data = pickle.load(f)
    with (dataset_path / "price_data.pkl").open("rb") as f:
        price_data = pickle.load(f)
    return (
        np.asarray(eod_data, dtype=np.float32),
        np.asarray(mask_data, dtype=np.float32),
        np.asarray(gt_data, dtype=np.float32),
        np.asarray(price_data, dtype=np.float32),
    )


def evaluate(prediction, ground_truth, mask):
    assert ground_truth.shape == prediction.shape, "shape mismatch"
    performance = {}
    performance["mse"] = np.linalg.norm((prediction - ground_truth) * mask) ** 2 / np.sum(mask)
    df_pred = pd.DataFrame(prediction * mask)
    df_gt = pd.DataFrame(ground_truth * mask)
    ic = []
    sharpe_li5 = []
    prec_10 = []
    for i in range(prediction.shape[1]):
        ic.append(df_pred[i].corr(df_gt[i]))
        rank_gt = np.argsort(ground_truth[:, i])
        gt_top10 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(gt_top10) < 10:
                gt_top10.add(cur_rank)

        rank_pre = np.argsort(prediction[:, i])
        pre_top5 = set()
        pre_top10 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_pre[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(pre_top5) < 5:
                pre_top5.add(cur_rank)
            if len(pre_top10) < 10:
                pre_top10.add(cur_rank)

        real_ret_rat_top5 = sum(ground_truth[pre][i] for pre in pre_top5) / 5
        prec = 0.0
        for pre in pre_top10:
            prec += ground_truth[pre][i] >= 0
        prec_10.append(prec / 10)
        sharpe_li5.append(real_ret_rat_top5)

    ic = np.asarray(ic, dtype=np.float64)
    sharpe_li5 = np.asarray(sharpe_li5, dtype=np.float64)
    performance["IC"] = float(np.nanmean(ic))
    performance["RIC"] = float(np.nanmean(ic) / np.nanstd(ic))
    performance["sharpe5"] = float((np.mean(sharpe_li5) / np.std(sharpe_li5)) * 15.87)
    performance["prec_10"] = float(np.mean(prec_10))
    return performance


def format_metrics(perf):
    return "mse:{mse:.4e}, IC:{IC:.4e}, RIC:{RIC:.4e}, prec@10:{prec_10:.4e}, SR:{sharpe5:.4e}".format(**perf)


class Trainer:
    def __init__(self, args, config, device, data, context_bundle):
        self.args = args
        self.config = config
        self.device = device
        self.eod_data, self.mask_data, self.gt_data, self.price_data = data
        self.context_bundle = context_bundle
        self.trade_dates = self.mask_data.shape[1]
        self.train_offsets = np.arange(
            0,
            config["valid_index"] - args.lookback_length - args.steps + 1,
            dtype=int,
        )
        self.model = self.build_model().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)

    def build_model(self):
        kwargs = dict(
            stocks=self.config["stock_num"],
            time_steps=self.args.lookback_length,
            channels=self.args.fea_num,
            market=self.config["market_num"],
            scale=self.args.scale_factor,
            activation=self.args.activation,
            main_mixer_activation=self.args.main_mixer_activation,
            scale_mixer_activation=self.args.scale_mixer_activation,
        )
        if self.args.model == "stockmixer":
            return StockMixer(**kwargs, stock_activation=self.args.stock_activation)
        return CAGStockMixer(
            **kwargs,
            depth=self.args.cag_depth,
            context_dim=len(FEATURE_NAMES),
            dropout=self.args.cag_dropout,
        )

    def get_batch(self, offset):
        seq_len = self.args.lookback_length
        steps = self.args.steps
        mask_batch = self.mask_data[:, offset : offset + seq_len + steps]
        mask_batch = np.min(mask_batch, axis=1)
        batch = (
            self.eod_data[:, offset : offset + seq_len, :],
            np.expand_dims(mask_batch, axis=1),
            np.expand_dims(self.price_data[:, offset + seq_len - 1], axis=1),
            np.expand_dims(self.gt_data[:, offset + seq_len + steps - 1], axis=1),
            self.context_bundle.normalized[offset],
        )
        return batch

    def to_device(self, batch):
        return tuple(torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in batch)

    def forward(self, data_batch, context_batch):
        if self.args.model == "stockmixer":
            return self.model(data_batch)
        return self.model(data_batch, context_batch)

    def validate(self, start_index, end_index, return_arrays=False):
        stock_num = self.config["stock_num"]
        with torch.no_grad():
            cur_pred = np.zeros([stock_num, end_index - start_index], dtype=np.float32)
            cur_gt = np.zeros([stock_num, end_index - start_index], dtype=np.float32)
            cur_mask = np.zeros([stock_num, end_index - start_index], dtype=np.float32)
            raw_ctx = np.zeros([end_index - start_index, len(FEATURE_NAMES)], dtype=np.float32)
            norm_ctx = np.zeros_like(raw_ctx)
            loss = 0.0
            reg_loss = 0.0
            rank_loss = 0.0
            offset_start = start_index - self.args.lookback_length - self.args.steps + 1
            offset_end = end_index - self.args.lookback_length - self.args.steps + 1
            for cur_offset in range(offset_start, offset_end):
                data_batch, mask_batch, price_batch, gt_batch, context_batch = self.to_device(self.get_batch(cur_offset))
                prediction = self.forward(data_batch, context_batch)
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
                cur_pred[:, result_offset] = cur_rr[:, 0].detach().cpu().numpy()
                cur_gt[:, result_offset] = gt_batch[:, 0].detach().cpu().numpy()
                cur_mask[:, result_offset] = mask_batch[:, 0].detach().cpu().numpy()
                raw_ctx[result_offset] = self.context_bundle.raw[cur_offset]
                norm_ctx[result_offset] = self.context_bundle.normalized[cur_offset]
            denom = max(end_index - start_index, 1)
            perf = evaluate(cur_pred, cur_gt, cur_mask)
            result = (loss / denom, reg_loss / denom, rank_loss / denom, perf)
            if return_arrays:
                return result + (cur_pred, cur_gt, cur_mask, raw_ctx, norm_ctx)
            return result

    def train(self):
        stock_num = self.config["stock_num"]
        valid_index = self.config["valid_index"]
        test_index = self.config["test_index"]
        best_valid_loss = np.inf
        best_epoch = None
        best_state = None
        rows = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            np.random.shuffle(self.train_offsets)
            self.model.train()
            tra_loss = 0.0
            tra_reg_loss = 0.0
            tra_rank_loss = 0.0
            for offset in self.train_offsets:
                data_batch, mask_batch, price_batch, gt_batch, context_batch = self.to_device(self.get_batch(int(offset)))
                self.optimizer.zero_grad()
                prediction = self.forward(data_batch, context_batch)
                cur_loss, cur_reg_loss, cur_rank_loss, _ = get_loss(
                    prediction,
                    gt_batch,
                    price_batch,
                    mask_batch,
                    stock_num,
                    self.args.alpha,
                )
                cur_loss.backward()
                self.optimizer.step()
                tra_loss += cur_loss.item()
                tra_reg_loss += cur_reg_loss.item()
                tra_rank_loss += cur_rank_loss.item()

            train_denom = len(self.train_offsets)
            tra_loss /= train_denom
            tra_reg_loss /= train_denom
            tra_rank_loss /= train_denom
            self.model.eval()
            val_loss, val_reg_loss, val_rank_loss, val_perf = self.validate(valid_index, test_index)
            test_loss, test_reg_loss, test_rank_loss, test_perf = self.validate(test_index, self.trade_dates)
            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

            row = {
                "epoch": epoch,
                "train_loss": tra_loss,
                "train_reg_loss": tra_reg_loss,
                "train_rank_loss": tra_rank_loss,
                "valid_loss": val_loss,
                "valid_reg_loss": val_reg_loss,
                "valid_rank_loss": val_rank_loss,
                "test_loss": test_loss,
                "test_reg_loss": test_reg_loss,
                "test_rank_loss": test_rank_loss,
            }
            for prefix, perf in (("valid", val_perf), ("test", test_perf)):
                for name, value in perf.items():
                    row[f"{prefix}_{name}"] = value
            rows.append(row)
            print(
                f"epoch {epoch:03d} | train {tra_loss:.4e} | valid {val_loss:.4e} "
                f"| test {test_loss:.4e} | test {format_metrics(test_perf)}"
            )

        if best_state is None:
            raise RuntimeError("No best state was selected.")
        self.model.load_state_dict(best_state)
        self.model.eval()
        best_valid = self.validate(valid_index, test_index, return_arrays=True)
        best_test = self.validate(test_index, self.trade_dates, return_arrays=True)
        elapsed = time.time() - start_time
        return {
            "epoch_rows": rows,
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid[0],
            "best_valid_perf": best_valid[3],
            "best_test_loss": best_test[0],
            "best_test_perf": best_test[3],
            "best_test_arrays": best_test[4:],
            "elapsed_seconds": elapsed,
            "state_dict": best_state,
        }


def write_epoch_metrics(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)


def write_predictions(args, config, arrays, path):
    pred, gt, mask, raw_ctx, norm_ctx = arrays
    offset_start = config["test_index"] - args.lookback_length - args.steps + 1
    records = []
    for day_col in range(pred.shape[1]):
        offset = offset_start + day_col
        target_date_index = offset + args.lookback_length + args.steps - 1
        day_context = {
            f"raw_{name}": raw_ctx[day_col, i] for i, name in enumerate(FEATURE_NAMES)
        }
        day_context.update(
            {f"norm_{name}": norm_ctx[day_col, i] for i, name in enumerate(FEATURE_NAMES)}
        )
        for stock_id in range(pred.shape[0]):
            row = {
                "model": args.model,
                "market": args.market,
                "stock_id": stock_id,
                "target_date_index": target_date_index,
                "offset": offset,
                "prediction_return": pred[stock_id, day_col],
                "target_return": gt[stock_id, day_col],
                "mask": mask[stock_id, day_col],
            }
            row.update(day_context)
            records.append(row)
    pd.DataFrame.from_records(records).to_csv(path, index=False)


def append_metadata(args, config, device_name, trainer, result, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "run_timestamp": int(time.time()),
        "model": args.model,
        "market": args.market,
        "epochs": args.epochs,
        "numpy_seed": args.numpy_seed,
        "torch_seed": args.torch_seed,
        "device": device_name,
        "lookback_length": args.lookback_length,
        "steps": args.steps,
        "alpha": args.alpha,
        "learning_rate": args.learning_rate,
        "activation": args.activation,
        "main_mixer_activation": args.main_mixer_activation,
        "scale_mixer_activation": args.scale_mixer_activation,
        "stock_activation": args.stock_activation,
        "stock_num": config["stock_num"],
        "valid_index": config["valid_index"],
        "test_index": config["test_index"],
        "train_offset_policy": "no_leakage_clean",
        "train_offset_start": int(np.min(trainer.train_offsets)),
        "train_offset_end": int(np.max(trainer.train_offsets)),
        "context_fit_split": trainer.context_bundle.fit_split,
        "context_normalization": trainer.context_bundle.normalization,
        "context_train_offset_start": trainer.context_bundle.train_offset_start,
        "context_train_offset_end": trainer.context_bundle.train_offset_end,
        "cag_depth": args.cag_depth if args.model == "cag_gated_context" else 0,
        "cag_dropout": args.cag_dropout if args.model == "cag_gated_context" else 0.0,
        "param_count": parameter_count(trainer.model),
        "best_epoch": result["best_epoch"],
        "best_valid_loss": result["best_valid_loss"],
        "best_test_loss": result["best_test_loss"],
        "elapsed_seconds": result["elapsed_seconds"],
    }
    for prefix, perf in (("best_valid", result["best_valid_perf"]), ("best_test", result["best_test_perf"])):
        for name, value in perf.items():
            row[f"{prefix}_{name}"] = value
    for name, value in PAPER_REFERENCE.get(args.market, {}).items():
        row[name] = value

    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    device, device_name = get_device(args.require_gpu)
    set_seeds(args.numpy_seed, args.torch_seed)
    config = MARKET_CONFIGS[args.market]
    data = load_market_data(args.dataset_root, args.market, args.steps)
    args.fea_num = data[0].shape[2]
    context_bundle = build_context_bundle(
        data[0],
        data[1],
        valid_index=config["valid_index"],
        lookback=args.lookback_length,
        steps=args.steps,
    )
    save_context_outputs(context_bundle, args.results_dir)
    trainer = Trainer(args, config, device, data, context_bundle)
    print(f"Using device: {device} ({device_name})")
    print(f"Model: {args.model} | params: {parameter_count(trainer.model)}")
    print(f"Dataset root: {args.dataset_root.resolve()}")
    print(
        f"Context scaler fit split: {context_bundle.fit_split} "
        f"offsets {context_bundle.train_offset_start}-{context_bundle.train_offset_end}"
    )
    result = trainer.train()

    stem = f"{args.model}_{args.market}_seed0"
    write_epoch_metrics(result["epoch_rows"], args.results_dir / f"epoch_metrics_{stem}.csv")
    write_predictions(args, config, result["best_test_arrays"], args.results_dir / f"preds_{stem}.csv")
    append_metadata(args, config, device_name, trainer, result, args.results_dir / "run_metadata.csv")
    if args.save_checkpoint:
        torch.save(
            {
                "model_state_dict": result["state_dict"],
                "args": vars(args),
                "best_epoch": result["best_epoch"],
                "best_valid_loss": result["best_valid_loss"],
            },
            args.results_dir / f"checkpoint_{stem}.pt",
        )
    print(f"Best epoch: {result['best_epoch']}")
    print(f"Best valid: {format_metrics(result['best_valid_perf'])}")
    print(f"Best test: {format_metrics(result['best_test_perf'])}")


if __name__ == "__main__":
    main()
