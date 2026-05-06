import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import torch

from evaluator import evaluate
from model import StockMixer, get_loss


PAPER_NASDAQ = {
    "IC": 0.043,
    "RIC": 0.501,
    "prec_10": 0.545,
    "sharpe5": 1.465,
}


ACTIVATIONS = ("hardswish", "relu", "gelu")
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


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train StockMixer and compare NASDAQ metrics with the paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--market", default="NASDAQ", choices=sorted(MARKET_CONFIGS))
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
    parser.add_argument(
        "--require-gpu",
        default="RTX 3090",
        help="Required substring in cuda:0 device name. Use an empty value to disable.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "dataset",
        help="Path to the dataset directory. Defaults to the repo-local dataset folder.",
    )
    parser.add_argument("--lookback-length", type=int, default=16)
    parser.add_argument("--fea-num", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--scale-factor", type=int, default=3)
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
    parser.add_argument("--numpy-seed", type=int, default=123456789)
    parser.add_argument("--torch-seed", type=int, default=12345678)
    args = parser.parse_args()
    return resolve_args(parser, args)


def resolve_args(parser, args):
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
    if args.main_mixer_activation is None:
        args.main_mixer_activation = args.activation
    if args.scale_mixer_activation is None:
        args.scale_mixer_activation = args.activation
    return args


def get_device(required_gpu):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this run requires cuda:0.")

    torch.cuda.set_device(0)
    device_name = torch.cuda.get_device_name(0)
    if required_gpu and required_gpu not in device_name:
        raise RuntimeError(
            "cuda:0 is '{}', expected a GPU name containing '{}'.".format(
                device_name, required_gpu
            )
        )
    return torch.device("cuda:0"), device_name


def set_seeds(numpy_seed, torch_seed):
    random.seed(numpy_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
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


class Trainer:
    def __init__(self, args, config, device, data):
        self.args = args
        self.config = config
        self.device = device
        self.eod_data, self.mask_data, self.gt_data, self.price_data = data
        self.trade_dates = self.mask_data.shape[1]
        self.batch_offsets = np.arange(start=0, stop=config["valid_index"], dtype=int)

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
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)

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

    def validate(self, start_index, end_index):
        stock_num = self.config["stock_num"]
        with torch.no_grad():
            cur_valid_pred = np.zeros([stock_num, end_index - start_index], dtype=float)
            cur_valid_gt = np.zeros([stock_num, end_index - start_index], dtype=float)
            cur_valid_mask = np.zeros([stock_num, end_index - start_index], dtype=float)
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
            loss = loss / (end_index - start_index)
            reg_loss = reg_loss / (end_index - start_index)
            rank_loss = rank_loss / (end_index - start_index)
            cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
        return loss, reg_loss, rank_loss, cur_valid_perf

    def train(self):
        stock_num = self.config["stock_num"]
        valid_index = self.config["valid_index"]
        test_index = self.config["test_index"]
        train_steps = valid_index - self.args.lookback_length - self.args.steps + 1
        best_valid_loss = np.inf
        best_epoch = None
        best_valid_perf = None
        best_test_perf = None

        for epoch in range(self.args.epochs):
            print("epoch{}##########################################################".format(epoch + 1))
            np.random.shuffle(self.batch_offsets)
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
            tra_loss = tra_loss / train_steps
            tra_reg_loss = tra_reg_loss / train_steps
            tra_rank_loss = tra_rank_loss / train_steps
            print(
                "Train : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}".format(
                    tra_loss, tra_reg_loss, tra_rank_loss
                )
            )

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

            print("Valid performance:\n", format_metrics(val_perf))
            print("Test performance:\n", format_metrics(test_perf), "\n")

        return {
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid_loss,
            "best_valid_perf": best_valid_perf,
            "best_test_perf": best_test_perf,
        }


def run_once(args, run_index, device, data):
    numpy_seed = args.numpy_seed + run_index
    torch_seed = args.torch_seed + run_index
    set_seeds(numpy_seed, torch_seed)

    config = MARKET_CONFIGS[args.market]
    trainer = Trainer(args, config, device, data)
    print(
        (
            "Run {}/{} | market={} | activation={} | main={} | scale={} | stock={} | "
            "epochs={} | epoch_preset={} | numpy_seed={} | torch_seed={}"
        ).format(
            run_index + 1,
            args.runs,
            args.market,
            args.activation,
            args.main_mixer_activation,
            args.scale_mixer_activation,
            args.stock_activation,
            args.epochs,
            args.epoch_preset,
            numpy_seed,
            torch_seed,
        )
    )
    return trainer.train()


def main():
    args = parse_args()
    device, device_name = get_device(args.require_gpu)
    print("Using device: {} ({})".format(device, device_name))
    print("Dataset root: {}".format(args.dataset_root.resolve()))

    data = load_market_data(args.dataset_root, args.market, args.steps)
    results = []
    for run_index in range(args.runs):
        result = run_once(args, run_index, device, data)
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
