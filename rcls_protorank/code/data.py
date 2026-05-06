import pickle
import random
from pathlib import Path

import numpy as np
import torch


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


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_market_data(dataset_root, market_name, steps=1):
    dataset_root = Path(dataset_root).resolve()
    market_name = market_name.upper()
    if market_name == "SP500":
        data = np.load(dataset_root / "SP500" / "SP500.npy")
        data = data[:, 915:, :]
        price_data = data[:, :, -1]
        mask_data = np.ones((data.shape[0], data.shape[1]), dtype=np.float32)
        eod_data = data.astype(np.float32)
        gt_data = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
        for stock_idx in range(data.shape[0]):
            for row in range(steps, data.shape[1]):
                prev = data[stock_idx][row - steps][-1]
                if abs(prev) > 1e-12:
                    gt_data[stock_idx][row] = (data[stock_idx][row][-1] - prev) / prev
        return eod_data, mask_data, gt_data, price_data.astype(np.float32)

    dataset_path = dataset_root / market_name
    with open(dataset_path / "eod_data.pkl", "rb") as f:
        eod_data = pickle.load(f)
    with open(dataset_path / "mask_data.pkl", "rb") as f:
        mask_data = pickle.load(f)
    with open(dataset_path / "gt_data.pkl", "rb") as f:
        gt_data = pickle.load(f)
    with open(dataset_path / "price_data.pkl", "rb") as f:
        price_data = pickle.load(f)
    return (
        np.asarray(eod_data, dtype=np.float32),
        np.asarray(mask_data, dtype=np.float32),
        np.asarray(gt_data, dtype=np.float32),
        np.asarray(price_data, dtype=np.float32),
    )


def compute_offsets(num_days, valid_index, test_index, lookback, steps):
    train_end = valid_index - lookback - steps + 1
    valid_start = valid_index - lookback - steps + 1
    valid_end = test_index - lookback - steps + 1
    test_start = test_index - lookback - steps + 1
    test_end = num_days - lookback - steps + 1
    if min(train_end, valid_end - valid_start, test_end - test_start) <= 0:
        raise ValueError("Invalid split/offset configuration.")
    return {
        "train": np.arange(0, train_end, dtype=int),
        "valid": np.arange(valid_start, valid_end, dtype=int),
        "test": np.arange(test_start, test_end, dtype=int),
    }


def target_day(offset, lookback, steps):
    return int(offset + lookback + steps - 1)


def get_batch(eod_data, mask_data, gt_data, price_data, offset, lookback, steps):
    mask_window = mask_data[:, offset : offset + lookback + steps]
    mask_batch = np.min(mask_window, axis=1, keepdims=True).astype(np.float32)
    x = eod_data[:, offset : offset + lookback, :].astype(np.float32)
    price = price_data[:, offset + lookback - 1 : offset + lookback].astype(np.float32)
    target = gt_data[
        :,
        offset + lookback + steps - 1 : offset + lookback + steps,
    ].astype(np.float32)
    return x, mask_batch, price, target


def lookback_regime_feature_dict(price_window, mask=None, eps=1e-6):
    prices = np.asarray(price_window, dtype=np.float64)
    if prices.ndim != 2 or prices.shape[1] < 2:
        return {name: np.nan for name in REGIME_FEATURE_NAMES}
    prices = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)
    ret = (prices[:, 1:] - prices[:, :-1]) / (np.abs(prices[:, :-1]) + eps)
    ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
    ret = np.clip(ret, -0.5, 0.5)
    if mask is not None:
        row_mask = np.asarray(mask).reshape(-1) > 0.5
        if np.any(row_mask):
            ret = ret[row_mask]
    market_ret = ret.mean(axis=0)
    downside = np.minimum(market_ret, 0.0)
    stock_sign = np.sign(ret)
    market_sign = np.sign(market_ret)[None, :]
    return {
        "market_ret_mean": float(market_ret.mean()),
        "market_ret_std": float(market_ret.std()),
        "market_ret_last": float(market_ret[-1]),
        "downside_vol": float(np.sqrt(np.mean(downside ** 2) + eps)),
        "dispersion": float(ret.std(axis=0).mean()),
        "synchronism": float((stock_sign == market_sign).mean()),
        "mean_abs_ret": float(np.abs(ret).mean()),
        "max_abs_ret": float(np.abs(ret).max()),
        "frac_positive": float((ret > 0).mean()),
    }

