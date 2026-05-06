from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


FEATURE_NAMES = [
    "mean_return",
    "momentum_slope",
    "realized_volatility",
    "dispersion",
    "pca_ratio",
]


@dataclass
class ContextBundle:
    raw: np.ndarray
    normalized: np.ndarray
    train_min: np.ndarray
    train_max: np.ndarray
    train_offset_start: int
    train_offset_end: int
    fit_split: str = "train"
    normalization: str = "train_minmax_clipped"

    def stats_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["raw"] = {
            name: {
                "min": float(np.nanmin(self.raw[:, i])),
                "max": float(np.nanmax(self.raw[:, i])),
                "mean": float(np.nanmean(self.raw[:, i])),
                "std": float(np.nanstd(self.raw[:, i])),
            }
            for i, name in enumerate(FEATURE_NAMES)
        }
        data["normalized"] = {
            name: {
                "min": float(np.nanmin(self.normalized[:, i])),
                "max": float(np.nanmax(self.normalized[:, i])),
                "mean": float(np.nanmean(self.normalized[:, i])),
                "std": float(np.nanstd(self.normalized[:, i])),
            }
            for i, name in enumerate(FEATURE_NAMES)
        }
        data["train_min"] = {
            name: float(self.train_min[i]) for i, name in enumerate(FEATURE_NAMES)
        }
        data["train_max"] = {
            name: float(self.train_max[i]) for i, name in enumerate(FEATURE_NAMES)
        }
        return data


def _safe_log_returns(closes: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    closes = np.asarray(closes, dtype=np.float64)
    closes = np.where(np.isfinite(closes) & (closes > eps), closes, np.nan)
    log_px = np.log(closes)
    return np.diff(log_px, axis=1)


def _pca_ratio_from_returns(returns: np.ndarray) -> float:
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


def _context_for_offset(
    eod_data: np.ndarray,
    mask_data: np.ndarray,
    offset: int,
    lookback: int,
    close_col: int,
) -> np.ndarray:
    closes = eod_data[:, offset : offset + lookback, close_col]
    valid_stock = np.min(mask_data[:, offset : offset + lookback], axis=1) > 0.5
    closes = closes[valid_stock]
    returns = _safe_log_returns(closes)
    finite_stock = np.isfinite(returns).mean(axis=1) >= 0.8
    returns = returns[finite_stock]
    if returns.size == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    index_series = returns.mean(axis=0)
    x = np.arange(index_series.shape[0], dtype=np.float64)
    x_var = float(x.var())
    if x_var <= 1e-20:
        slope = 0.0
    else:
        slope = float(((x - x.mean()) * (index_series - index_series.mean())).mean() / x_var)

    features = np.array(
        [
            float(returns.mean()),
            slope,
            float(index_series.std()),
            float(returns.std(axis=0).mean()),
            _pca_ratio_from_returns(returns),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def build_context_bundle(
    eod_data: np.ndarray,
    mask_data: np.ndarray,
    valid_index: int,
    lookback: int,
    steps: int,
    close_col: int = -1,
) -> ContextBundle:
    trade_dates = mask_data.shape[1]
    max_offset = trade_dates - lookback - steps
    if max_offset < 0:
        raise ValueError("Not enough trade dates for the requested lookback and steps.")

    raw = np.zeros((max_offset + 1, len(FEATURE_NAMES)), dtype=np.float32)
    for offset in range(max_offset + 1):
        raw[offset] = _context_for_offset(eod_data, mask_data, offset, lookback, close_col)

    train_offset_start = 0
    train_offset_end = valid_index - lookback - steps
    if train_offset_end < train_offset_start:
        raise ValueError("Invalid train offset range for context scaler.")

    train_raw = raw[train_offset_start : train_offset_end + 1]
    train_min = np.nanmin(train_raw, axis=0)
    train_max = np.nanmax(train_raw, axis=0)
    denom = np.maximum(train_max - train_min, 1e-12)
    normalized = np.clip((raw - train_min) / denom, 0.0, 1.0).astype(np.float32)

    return ContextBundle(
        raw=raw,
        normalized=normalized,
        train_min=train_min.astype(np.float32),
        train_max=train_max.astype(np.float32),
        train_offset_start=train_offset_start,
        train_offset_end=train_offset_end,
    )


def save_context_outputs(bundle: ContextBundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "context_stats.json").open("w", encoding="utf-8") as f:
        json.dump(bundle.stats_dict(), f, indent=2)

    rows = np.concatenate([bundle.raw, bundle.normalized], axis=1)
    header = ",".join(
        [f"raw_{name}" for name in FEATURE_NAMES]
        + [f"norm_{name}" for name in FEATURE_NAMES]
    )
    np.savetxt(
        output_dir / "context_stats.csv",
        rows,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.10g",
    )


def summarize_context_for_offset(bundle: ContextBundle, offset: int) -> Tuple[np.ndarray, np.ndarray]:
    return bundle.raw[offset], bundle.normalized[offset]
