from __future__ import annotations

import numpy as np
import pandas as pd


def _valid_xy(pred, target, mask=None):
    pred = np.asarray(pred).reshape(-1)
    target = np.asarray(target).reshape(-1)
    ok = np.isfinite(pred) & np.isfinite(target)
    if mask is not None:
        ok &= np.asarray(mask).reshape(-1) > 0.5
    return pred[ok], target[ok]


def pearson_ic(pred, target, mask=None):
    x, y = _valid_xy(pred, target, mask)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def rank_ic(pred, target, mask=None):
    x, y = _valid_xy(pred, target, mask)
    if len(x) < 3:
        return np.nan
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def precision_at_k(pred, target, k=10, mask=None):
    x, y = _valid_xy(pred, target, mask)
    if len(x) == 0:
        return np.nan
    k = min(k, len(x))
    idx = np.argsort(-x)[:k]
    return float(np.mean(y[idx] > 0))


def topk_return(pred, target, k=10, mask=None):
    x, y = _valid_xy(pred, target, mask)
    if len(x) == 0:
        return np.nan
    k = min(k, len(x))
    idx = np.argsort(-x)[:k]
    return float(np.mean(y[idx]))


def long_short_return(pred, target, k=10, mask=None):
    x, y = _valid_xy(pred, target, mask)
    if len(x) < 2:
        return np.nan
    k = min(k, len(x) // 2)
    if k == 0:
        return np.nan
    long_idx = np.argsort(-x)[:k]
    short_idx = np.argsort(x)[:k]
    return float(np.mean(y[long_idx]) - np.mean(y[short_idx]))


def masked_mae(pred, target, mask=None):
    x, y = _valid_xy(pred, target, mask)
    if len(x) == 0:
        return np.nan
    return float(np.mean(np.abs(x - y)))


def aggregate_daily_metrics(df, pred_col="pred", target_col="target", mask_col="mask"):
    rows = []
    for day, group in df.groupby("day_idx", sort=True):
        mask = group[mask_col].values if mask_col in group else None
        pred = group[pred_col].values
        target = group[target_col].values
        rows.append(
            {
                "day_idx": day,
                "ic": pearson_ic(pred, target, mask),
                "rankic": rank_ic(pred, target, mask),
                "p10": precision_at_k(pred, target, 10, mask),
                "p20": precision_at_k(pred, target, 20, mask),
                "top10_return": topk_return(pred, target, 10, mask),
                "top20_return": topk_return(pred, target, 20, mask),
                "long_short10": long_short_return(pred, target, 10, mask),
                "mae": masked_mae(pred, target, mask),
            }
        )
    daily = pd.DataFrame(rows)
    if daily.empty:
        return {
            "ic": np.nan,
            "rankic": np.nan,
            "icir": np.nan,
            "p10": np.nan,
            "p20": np.nan,
            "top10_return": np.nan,
            "top20_return": np.nan,
            "long_short10": np.nan,
            "sharpe_top10": np.nan,
            "sharpe_long_short10": np.nan,
            "mae": np.nan,
            "num_days": 0,
        }, daily

    def mean(name):
        return float(daily[name].mean(skipna=True))

    def sharpe(name):
        values = daily[name].dropna().to_numpy(dtype=float)
        if len(values) < 2 or np.std(values) < 1e-12:
            return np.nan
        return float(np.mean(values) / (np.std(values) + 1e-12))

    ic_values = daily["ic"].dropna().to_numpy(dtype=float)
    out = {
        "ic": mean("ic"),
        "rankic": mean("rankic"),
        "icir": float(np.mean(ic_values) / (np.std(ic_values) + 1e-12)) if len(ic_values) else np.nan,
        "p10": mean("p10"),
        "p20": mean("p20"),
        "top10_return": mean("top10_return"),
        "top20_return": mean("top20_return"),
        "long_short10": mean("long_short10"),
        "sharpe_top10": sharpe("top10_return"),
        "sharpe_long_short10": sharpe("long_short10"),
        "mae": mean("mae"),
        "num_days": int(len(daily)),
    }
    return out, daily


def metric_row_from_predictions(df, dataset=None, model=None, seed=None, split=None):
    metrics, _ = aggregate_daily_metrics(df)
    row = {
        "dataset": dataset if dataset is not None else df.get("dataset", pd.Series([""])).iloc[0],
        "model": model if model is not None else df.get("model", pd.Series([""])).iloc[0],
        "seed": seed if seed is not None else df.get("seed", pd.Series([0])).iloc[0],
        "split": split if split is not None else df.get("split", pd.Series([""])).iloc[0],
    }
    row.update(metrics)
    return row
