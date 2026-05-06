import argparse
import csv
import pickle
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from evaluator import evaluate
from model import StockMixer, get_loss


BASE_NUMPY_SEED = 123456789
BASE_TORCH_SEED = 12345678

PAPER_NASDAQ = {
    "IC": 0.043,
    "RIC": 0.501,
    "prec_10": 0.545,
    "sharpe5": 1.465,
}

ACTIVATIONS = ("hardswish", "relu", "gelu")
MODEL_CHOICES = ("stockmixer", "rcls_f_k3", "rcls_f_k1")
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
    "stress_source",
    "regime_0",
    "regime_1",
    "regime_2",
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
]


def parse_args():
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
    parser.add_argument("--lookback-length", type=int, default=16)
    parser.add_argument("--fea-num", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--scale-factor", type=int, default=3)
    parser.add_argument("--gate-hidden", type=int, default=128)
    parser.add_argument("--rcls-dropout", type=float, default=0.10)
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
    args = parser.parse_args()
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
    if args.main_mixer_activation is None:
        args.main_mixer_activation = args.activation
    if args.scale_mixer_activation is None:
        args.scale_mixer_activation = args.activation
    if args.numpy_seed is None:
        args.numpy_seed = BASE_NUMPY_SEED + args.seed
    if args.torch_seed is None:
        args.torch_seed = BASE_TORCH_SEED + args.seed
    args.save_predictions = not args.no_save_predictions
    return args


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
        self.batch_offsets = np.arange(start=0, stop=config["valid_index"], dtype=int)
        self.output_root = args.output_root.resolve()
        self.results_dir = self.output_root / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.infer_time_ms_per_day = 0.0

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
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

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
        seq_len = self.args.lookback_length
        prices = np.asarray(self.price_data[:, offset : offset + seq_len], dtype=float)
        if prices.shape[1] < 2:
            return "", "", ""
        prev = prices[:, :-1]
        nxt = prices[:, 1:]
        denom = np.where(np.abs(prev) > 1e-12, prev, np.nan)
        returns = (nxt - prev) / denom
        with np.errstate(invalid="ignore"):
            market_ret = np.nanmean(returns, axis=0)
            market_vol = np.nanstd(market_ret)
            dispersion = np.nanmean(np.nanstd(returns, axis=0))
            market_sign = np.sign(market_ret)[None, :]
            valid = np.isfinite(returns) & np.isfinite(market_sign)
            matches = np.where(valid, np.sign(returns) == market_sign, np.nan)
            synchronism = np.nanmean(matches)
        return safe_float(market_vol), safe_float(synchronism), safe_float(dispersion)

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

    def prediction_rows(self, split_name, cur_offset, result_offset, cur_rr, gt_batch, mask_batch):
        market_vol, synchronism, dispersion = self.lookback_stress_features(cur_offset)
        gate0, gate1, gate2 = self.gate_values()
        day_idx = cur_offset + self.args.lookback_length + self.args.steps - 1
        pred = cur_rr[:, 0].detach().cpu().numpy()
        target = gt_batch[:, 0].detach().cpu().numpy()
        mask = mask_batch[:, 0].detach().cpu().numpy()
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
                    "stress_source": "lookback",
                    "regime_0": gate0,
                    "regime_1": gate1,
                    "regime_2": gate2,
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
            loss = 0.0
            reg_loss = 0.0
            rank_loss = 0.0
            offset_start = start_index - self.args.lookback_length - self.args.steps + 1
            offset_end = end_index - self.args.lookback_length - self.args.steps + 1
            for cur_offset in range(offset_start, offset_end):
                data_batch, mask_batch, price_batch, gt_batch = self.to_device(
                    self.get_batch(cur_offset)
                )
                prediction = self.model(data_batch)
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
            denom = end_index - start_index
            loss = loss / denom
            reg_loss = reg_loss / denom
            rank_loss = rank_loss / denom
            cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
        elapsed = time.time() - eval_start
        return loss, reg_loss, rank_loss, cur_valid_perf, rows, elapsed

    def validate(self, start_index, end_index):
        loss, reg_loss, rank_loss, perf, _, _ = self.evaluate_range(start_index, end_index)
        return loss, reg_loss, rank_loss, perf

    def save_best_predictions(self):
        valid_index = self.config["valid_index"]
        test_index = self.config["test_index"]
        self.model.eval()
        _, _, _, _, valid_rows, _ = self.evaluate_range(
            valid_index, test_index, split_name="valid", collect_rows=True
        )
        _, _, _, _, test_rows, test_elapsed = self.evaluate_range(
            test_index, self.trade_dates, split_name="test", collect_rows=True
        )
        output = self.results_dir / "preds_{}_{}_seed{}.csv".format(
            self.args.model, self.args.market, self.result_seed
        )
        with output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=PREDICTION_COLUMNS)
            writer.writeheader()
            for row in valid_rows:
                writer.writerow(row)
            for row in test_rows:
                writer.writerow(row)
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
        }
        upsert_csv_row(
            self.results_dir / "run_metadata.csv",
            METADATA_COLUMNS,
            row,
            ["dataset", "model", "seed"],
        )

    def train(self):
        stock_num = self.config["stock_num"]
        valid_index = self.config["valid_index"]
        test_index = self.config["test_index"]
        train_steps = valid_index - self.args.lookback_length - self.args.steps + 1
        best_valid_loss = np.inf
        best_epoch = None
        best_valid_perf = None
        best_test_perf = None
        best_state = None
        epochs_without_improvement = 0
        train_time_sec = 0.0
        epochs_ran = 0

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        total_start = time.time()

        for epoch in range(self.args.epochs):
            epochs_ran = epoch + 1
            print("epoch{}##########################################################".format(epoch + 1))
            np.random.shuffle(self.batch_offsets)
            self.model.train()
            epoch_train_start = time.time()
            tra_loss = 0.0
            tra_reg_loss = 0.0
            tra_rank_loss = 0.0
            for j in range(train_steps):
                data_batch, mask_batch, price_batch, gt_batch = self.to_device(
                    self.get_batch(self.batch_offsets[j])
                )
                self.optimizer.zero_grad()
                prediction = self.model(data_batch)
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
            train_time_sec += time.time() - epoch_train_start
            tra_loss = tra_loss / train_steps
            tra_reg_loss = tra_reg_loss / train_steps
            tra_rank_loss = tra_rank_loss / train_steps
            print(
                "Train : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}".format(
                    tra_loss, tra_reg_loss, tra_rank_loss
                )
            )

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
    (args.output_root / "results").mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(parents=True, exist_ok=True)
    device, device_name = get_device(args.require_gpu)
    print("Using device: {} ({})".format(device, device_name))
    print("Dataset root: {}".format(args.dataset_root.resolve()))
    print("Output root: {}".format(args.output_root.resolve()))

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
