import numpy as np
import pandas as pd


METRIC_COLUMNS = [
    "num_days",
    "num_rows",
    "ic",
    "rankic",
    "p10",
    "p20",
    "long_short",
    "sharpe",
    "mae",
    "nll",
]


def safe_corr(a, b, method="pearson"):
    s1 = pd.Series(a)
    s2 = pd.Series(b)
    valid = s1.notna() & s2.notna()
    if valid.sum() < 3:
        return np.nan
    if s1[valid].nunique(dropna=True) < 2 or s2[valid].nunique(dropna=True) < 2:
        return np.nan
    return s1[valid].corr(s2[valid], method=method)


def precision_at_k(day_df, k):
    d = day_df.sort_values("pred", ascending=False).head(k)
    return float((d["target"] > 0).mean()) if len(d) else np.nan


def long_short_return(day_df, k=10):
    d = day_df.sort_values("pred", ascending=False)
    if len(d) < 2 * k:
        return np.nan
    return float(d.head(k)["target"].mean() - d.tail(k)["target"].mean())


def compute_metrics_from_frame(df, include_nll=True):
    if df.empty:
        return {name: np.nan for name in METRIC_COLUMNS}
    work = df.copy()
    work["mask"] = pd.to_numeric(work["mask"], errors="coerce").fillna(0.0)
    work = work[work["mask"] > 0.5]
    for col in ["pred", "target", "mu", "sigma", "nll"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["pred", "target"])
    if work.empty:
        return {name: np.nan for name in METRIC_COLUMNS}

    ic = []
    rankic = []
    p10 = []
    p20 = []
    ls = []
    for _, day_df in work.groupby("day_idx"):
        ic.append(safe_corr(day_df["pred"], day_df["target"], "pearson"))
        rankic.append(safe_corr(day_df["pred"], day_df["target"], "spearman"))
        p10.append(precision_at_k(day_df, 10))
        p20.append(precision_at_k(day_df, 20))
        ls.append(long_short_return(day_df, 10))

    ls = np.asarray(ls, dtype=float)
    sharpe = np.nan
    if np.isfinite(ls).sum() > 1 and np.nanstd(ls) > 0:
        sharpe = float(np.nanmean(ls) / np.nanstd(ls) * np.sqrt(252.0))
    nll = np.nan
    if include_nll and "nll" in work.columns:
        nll = float(np.nanmean(work["nll"]))
    return {
        "num_days": int(work["day_idx"].nunique()),
        "num_rows": int(len(work)),
        "ic": float(np.nanmean(ic)),
        "rankic": float(np.nanmean(rankic)),
        "p10": float(np.nanmean(p10)),
        "p20": float(np.nanmean(p20)),
        "long_short": float(np.nanmean(ls)),
        "sharpe": sharpe,
        "mae": float(np.mean(np.abs(work.get("mu", work["pred"]) - work["target"]))),
        "nll": nll,
    }


def compute_metrics_from_arrays(prediction, target, mask, sigma=None):
    rows = []
    num_days = prediction.shape[1]
    for day_idx in range(num_days):
        for stock_idx in range(prediction.shape[0]):
            row = {
                "day_idx": day_idx,
                "stock_idx": stock_idx,
                "pred": prediction[stock_idx, day_idx],
                "target": target[stock_idx, day_idx],
                "mask": mask[stock_idx, day_idx],
            }
            if sigma is not None:
                sig = max(float(sigma[stock_idx, day_idx]), 1e-6)
                err = abs(row["target"] - prediction[stock_idx, day_idx])
                row["sigma"] = sig
                row["nll"] = err / sig + np.log(sig)
            rows.append(row)
    return compute_metrics_from_frame(pd.DataFrame(rows))
