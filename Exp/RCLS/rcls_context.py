from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CONTEXT_FEATURES = [
    "mean_return",
    "momentum_slope",
    "realized_volatility",
    "dispersion",
    "pca_ratio",
    "synchronism",
    "downside_volatility",
    "mean_abs_return",
    "max_abs_return",
    "frac_positive",
]

FIVE_CONTEXT_FEATURES = CONTEXT_FEATURES[:5]


def _safe_log_returns(close, eps=1e-12):
    close = np.asarray(close, dtype=np.float64)
    close = np.where(np.isfinite(close) & (close > eps), close, np.nan)
    return np.diff(np.log(close), axis=1)


def _pca_ratio(returns):
    if returns.shape[0] < 2 or returns.shape[1] < 2:
        return 0.0
    centered = returns - np.nanmean(returns, axis=1, keepdims=True)
    centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)
    gram = centered.T @ centered
    total = float(np.trace(gram))
    if total <= 1e-20:
        return 0.0
    eigvals = np.linalg.eigvalsh(gram)
    return float(max(eigvals[-1], 0.0) / total)


def compute_context_features(x, mask=None, eps=1e-8, close_col=-1):
    x = np.asarray(x, dtype=np.float64)
    close = x[:, :, close_col]
    if mask is not None:
        mask_arr = np.asarray(mask)
        if mask_arr.ndim == 2:
            valid = np.min(mask_arr[:, : close.shape[1]], axis=1) > 0.5
        else:
            valid = mask_arr.reshape(-1) > 0.5
        close = close[valid]
    returns = _safe_log_returns(close, eps=eps)
    if returns.size == 0:
        return {name: 0.0 for name in CONTEXT_FEATURES}
    finite_stock = np.isfinite(returns).mean(axis=1) >= 0.8
    returns = returns[finite_stock]
    if returns.size == 0:
        return {name: 0.0 for name in CONTEXT_FEATURES}
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    market_ret = returns.mean(axis=0)
    x_axis = np.arange(market_ret.shape[0], dtype=np.float64)
    x_var = float(x_axis.var())
    if x_var <= 1e-20:
        slope = 0.0
    else:
        slope = float(((x_axis - x_axis.mean()) * (market_ret - market_ret.mean())).mean() / x_var)
    market_sign = np.sign(market_ret)
    synchronism = np.mean(np.sign(returns) == market_sign[None, :])
    downside = market_ret[market_ret < 0]
    return {
        "mean_return": float(returns.mean()),
        "momentum_slope": slope,
        "realized_volatility": float(market_ret.std()),
        "dispersion": float(returns.std(axis=0).mean()),
        "pca_ratio": _pca_ratio(returns),
        "synchronism": float(synchronism),
        "downside_volatility": float(downside.std()) if len(downside) else 0.0,
        "mean_abs_return": float(np.abs(returns).mean()),
        "max_abs_return": float(np.abs(returns).max()),
        "frac_positive": float(np.mean(returns > 0)),
    }


class ContextNormalizer:
    def __init__(self, feature_names=None, method="minmax", eps=1e-8):
        self.feature_names = list(feature_names or CONTEXT_FEATURES)
        self.method = method
        self.eps = eps
        self.center = None
        self.scale = None
        self.fit_split = "train"

    def fit(self, context_rows):
        arr = self._rows_to_array(context_rows)
        if self.method == "zscore":
            self.center = np.nanmean(arr, axis=0)
            self.scale = np.nanstd(arr, axis=0)
        elif self.method == "minmax":
            self.center = np.nanmin(arr, axis=0)
            self.scale = np.nanmax(arr, axis=0) - self.center
        else:
            raise ValueError(f"Unknown context normalization: {self.method}")
        self.scale = np.where(np.abs(self.scale) < self.eps, 1.0, self.scale)
        return self

    def _rows_to_array(self, rows):
        if isinstance(rows, pd.DataFrame):
            return rows[self.feature_names].to_numpy(dtype=float)
        return np.asarray([[row[name] for name in self.feature_names] for row in rows], dtype=float)

    def transform_array(self, arr):
        if self.center is None or self.scale is None:
            raise RuntimeError("ContextNormalizer is not fitted.")
        out = (np.asarray(arr, dtype=float) - self.center) / self.scale
        if self.method == "minmax":
            out = np.clip(out, 0.0, 1.0)
        return out.astype(np.float32)

    def transform_dict(self, ctx):
        arr = np.asarray([[ctx[name] for name in self.feature_names]], dtype=float)
        values = self.transform_array(arr)[0]
        return {name: float(values[i]) for i, name in enumerate(self.feature_names)}

    def state_dict(self):
        return {
            "feature_names": self.feature_names,
            "method": self.method,
            "center": self.center.tolist() if self.center is not None else None,
            "scale": self.scale.tolist() if self.scale is not None else None,
            "fit_split": self.fit_split,
        }

    def load_state_dict(self, state):
        self.feature_names = list(state["feature_names"])
        self.method = state["method"]
        self.center = np.asarray(state["center"], dtype=float)
        self.scale = np.asarray(state["scale"], dtype=float)
        self.fit_split = state.get("fit_split", "train")
        return self

    def save_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path):
        with Path(path).open("r", encoding="utf-8") as f:
            state = json.load(f)
        return cls().load_state_dict(state)


@dataclass
class PseudoRegimeStats:
    num_regimes: int
    stress_median: float
    stress_q70: float
    dispersion_q70: float


def context_feature_table(eod_data, mask_data, offsets_by_split, lookback, steps):
    rows = []
    for split, offsets in offsets_by_split.items():
        for offset in offsets:
            offset = int(offset)
            x = eod_data[:, offset : offset + lookback, :]
            mask = mask_data[:, offset : offset + lookback]
            ctx = compute_context_features(x, mask)
            row = {"split": split, "offset": offset, "day_idx": offset + lookback + steps - 1}
            row.update(ctx)
            rows.append(row)
    return pd.DataFrame(rows)


def fit_pseudo_regime_stats(train_context_df, num_regimes=2):
    stress = stress_score(train_context_df)
    return PseudoRegimeStats(
        num_regimes=int(num_regimes),
        stress_median=float(np.nanmedian(stress)),
        stress_q70=float(np.nanquantile(stress, 0.70)),
        dispersion_q70=float(np.nanquantile(train_context_df["ctx_dispersion"], 0.70)),
    )


def stress_score(df):
    return (
        df["ctx_realized_volatility"].astype(float)
        + df["ctx_dispersion"].astype(float)
        + np.abs(df["ctx_mean_return"].astype(float))
        + df["ctx_downside_volatility"].astype(float)
        + df["ctx_pca_ratio"].astype(float)
    )


def apply_pseudo_regime_labels(context_df, stats):
    df = context_df.copy()
    score = stress_score(df)
    if stats.num_regimes == 3:
        labels = np.zeros(len(df), dtype=int)
        labels[score > stats.stress_q70] = 1
        labels[(df["ctx_dispersion"].to_numpy(dtype=float) > stats.dispersion_q70) & (labels == 0)] = 2
        margin = np.maximum(score - stats.stress_q70, df["ctx_dispersion"].to_numpy(dtype=float) - stats.dispersion_q70)
    else:
        labels = (score > stats.stress_median).astype(int)
        margin = score - stats.stress_median
    df["stress_score"] = score
    df["regime_label"] = labels
    df["regime_margin"] = margin
    denom = np.nanstd(margin) + 1e-8
    df["regime_confidence"] = 1.0 / (1.0 + np.exp(-np.abs(margin) / denom))
    return df


def build_context_table(eod_data, mask_data, offsets_by_split, lookback, steps, normalization="minmax"):
    raw = context_feature_table(eod_data, mask_data, offsets_by_split, lookback, steps)
    normalizer = ContextNormalizer(CONTEXT_FEATURES, method=normalization).fit(raw[raw["split"] == "train"])
    norm_arr = normalizer.transform_array(raw[CONTEXT_FEATURES].to_numpy(dtype=float))
    norm = raw[["split", "offset", "day_idx"]].copy()
    for idx, name in enumerate(CONTEXT_FEATURES):
        norm[f"ctx_{name}"] = norm_arr[:, idx]
        raw[f"raw_ctx_{name}"] = raw[name]
    stats2 = fit_pseudo_regime_stats(norm[norm["split"] == "train"], 2)
    stats3 = fit_pseudo_regime_stats(norm[norm["split"] == "train"], 3)
    norm2 = apply_pseudo_regime_labels(norm, stats2)
    norm3 = apply_pseudo_regime_labels(norm, stats3)
    norm["regime_label_k2"] = norm2["regime_label"]
    norm["regime_confidence_k2"] = norm2["regime_confidence"]
    norm["regime_margin_k2"] = norm2["regime_margin"]
    norm["stress_score"] = norm2["stress_score"]
    norm["regime_label_k3"] = norm3["regime_label"]
    norm["regime_confidence_k3"] = norm3["regime_confidence"]
    norm["regime_margin_k3"] = norm3["regime_margin"]
    return norm, raw, normalizer, {"k2": stats2.__dict__, "k3": stats3.__dict__}
