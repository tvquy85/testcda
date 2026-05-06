import json
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch

from data import REGIME_FEATURE_NAMES, lookback_regime_feature_dict, target_day


REGIME_MODES = [
    "latent_current",
    "manual_breadth2",
    "manual_stress2",
    "pseudo_breadth2",
    "pseudo_stress2",
    "jump_stress2",
    "cluster_cov2",
    "pseudo_market3",
]

SCORE_COLUMNS = ["breadth_score", "stress_score", "corr_score"]
REGIME_DIAGNOSTIC_COLUMNS = SCORE_COLUMNS + ["regime_label", "regime_confidence", "regime_margin"]
MANUAL_REGIME_MODES = {"manual_breadth2", "manual_stress2", "cluster_cov2"}
PSEUDO_REGIME_MODES = {"pseudo_breadth2", "pseudo_stress2", "jump_stress2", "pseudo_market3"}


@dataclass
class RegimeArtifact:
    mode: str
    num_regimes: int
    feature_mean: Dict[str, float]
    feature_std: Dict[str, float]
    thresholds: Dict[str, float]
    centroids: Optional[np.ndarray]
    label_counts: Dict[int, int]
    fit_split: str = "train"
    jump_min_run: int = 3

    def to_metadata(self):
        return {
            "mode": self.mode,
            "num_regimes": self.num_regimes,
            "fit_split": self.fit_split,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "thresholds": self.thresholds,
            "centroids": self.centroids.tolist() if self.centroids is not None else None,
            "label_counts": {str(k): int(v) for k, v in self.label_counts.items()},
            "jump_min_run": self.jump_min_run,
        }


def _safe_std(values):
    value = float(np.nanstd(values))
    return value if np.isfinite(value) and value > 1e-8 else 1.0


def _zscore(df, artifact):
    out = {}
    for name in REGIME_FEATURE_NAMES:
        out[name] = (df[name].astype(float) - artifact.feature_mean[name]) / artifact.feature_std[name]
    return out


def _score_frame(df, artifact):
    z = _zscore(df, artifact)
    out = df.copy()
    out["breadth_score"] = (
        z["market_ret_mean"]
        + 0.50 * z["market_ret_last"]
        + 0.75 * z["frac_positive"]
        - 0.50 * z["downside_vol"]
        - 0.25 * z["market_ret_std"]
    )
    out["stress_score"] = (
        z["market_ret_std"]
        + z["downside_vol"]
        + 0.50 * z["max_abs_ret"]
        + 0.50 * z["synchronism"]
        + 0.25 * z["mean_abs_ret"]
        - 0.25 * z["market_ret_mean"]
    )
    out["corr_score"] = z["synchronism"] + 0.50 * z["market_ret_std"] - 0.50 * z["dispersion"]
    return out


def _online_min_run_smooth(labels, min_run=3):
    labels = [int(x) for x in labels]
    if not labels:
        return []
    current = labels[0]
    candidate = current
    count = 0
    smoothed = []
    for label in labels:
        if label == current:
            candidate = current
            count = 0
        elif label == candidate:
            count += 1
        else:
            candidate = label
            count = 1
        if candidate != current and count >= min_run:
            current = candidate
            count = 0
        smoothed.append(current)
    return smoothed


def _kmeans2(x, seed=0, max_iter=50):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    stress = x[:, 0]
    low = np.nanpercentile(stress, 25)
    high = np.nanpercentile(stress, 75)
    c0 = x[np.argmin(np.abs(stress - low))].copy()
    c1 = x[np.argmin(np.abs(stress - high))].copy()
    if np.allclose(c0, c1):
        idx = rng.choice(len(x), size=2, replace=False)
        c0, c1 = x[idx[0]].copy(), x[idx[1]].copy()
    centroids = np.stack([c0, c1], axis=0)
    labels = np.zeros(len(x), dtype=int)
    for _ in range(max_iter):
        dist = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
        new_labels = dist.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for k in range(2):
            if np.any(labels == k):
                centroids[k] = x[labels == k].mean(axis=0)
    if centroids[0, 0] > centroids[1, 0]:
        centroids = centroids[[1, 0]]
    return centroids


def compute_regime_feature_table(price_data, mask_data, offsets_by_split, lookback, steps):
    rows = []
    for split, offsets in offsets_by_split.items():
        for offset in offsets:
            offset = int(offset)
            mask_window = mask_data[:, offset : offset + lookback + steps]
            mask = np.min(mask_window, axis=1, keepdims=True).astype(np.float32)
            features = lookback_regime_feature_dict(price_data[:, offset : offset + lookback], mask)
            rows.append(
                {
                    "split": split,
                    "offset": offset,
                    "day_idx": target_day(offset, lookback, steps),
                    **features,
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "offset"]).reset_index(drop=True)


def fit_regime_artifact(feature_df, mode, seed=0, jump_min_run=3):
    if mode not in REGIME_MODES:
        raise ValueError(f"Unknown regime mode: {mode}")
    num_regimes = 3 if mode == "pseudo_market3" else 2
    train = feature_df[feature_df["split"] == "train"].copy()
    means = {name: float(train[name].mean()) for name in REGIME_FEATURE_NAMES}
    stds = {name: _safe_std(train[name].values) for name in REGIME_FEATURE_NAMES}
    artifact = RegimeArtifact(
        mode=mode,
        num_regimes=num_regimes,
        feature_mean=means,
        feature_std=stds,
        thresholds={},
        centroids=None,
        label_counts={},
        jump_min_run=jump_min_run,
    )
    scored = _score_frame(train, artifact)
    artifact.thresholds = {
        "breadth_median": float(scored["breadth_score"].median()),
        "stress_median": float(scored["stress_score"].median()),
        "market_low": float((scored["breadth_score"] - scored["stress_score"]).quantile(1.0 / 3.0)),
        "market_high": float((scored["breadth_score"] - scored["stress_score"]).quantile(2.0 / 3.0)),
    }
    if mode == "cluster_cov2":
        artifact.centroids = _kmeans2(scored[SCORE_COLUMNS].values, seed=seed)
    applied = apply_regime_artifact(feature_df, artifact)
    label_counts = applied[applied["split"] == "train"]["regime_label"].value_counts().to_dict()
    artifact.label_counts = {int(k): int(v) for k, v in label_counts.items()}
    return artifact


def _assign_labels(scored, artifact):
    mode = artifact.mode
    if mode in {"manual_breadth2", "pseudo_breadth2"}:
        return (scored["breadth_score"] < artifact.thresholds["breadth_median"]).astype(int).values
    if mode in {"manual_stress2", "pseudo_stress2"}:
        return (scored["stress_score"] >= artifact.thresholds["stress_median"]).astype(int).values
    if mode == "jump_stress2":
        raw = (scored["stress_score"] >= artifact.thresholds["stress_median"]).astype(int).values
        labels = np.zeros_like(raw)
        for _, idx in scored.groupby("split", sort=False).groups.items():
            loc = np.asarray(list(idx), dtype=int)
            labels[loc] = _online_min_run_smooth(raw[loc], artifact.jump_min_run)
        return labels
    if mode == "cluster_cov2":
        x = scored[SCORE_COLUMNS].values
        dist = ((x[:, None, :] - artifact.centroids[None, :, :]) ** 2).sum(axis=-1)
        return dist.argmin(axis=1).astype(int)
    if mode == "pseudo_market3":
        score = scored["breadth_score"] - scored["stress_score"]
        labels = np.ones(len(scored), dtype=int)
        labels[score >= artifact.thresholds["market_high"]] = 0
        labels[score <= artifact.thresholds["market_low"]] = 2
        return labels
    return np.zeros(len(scored), dtype=int)


def _confidence(scored, labels, artifact):
    mode = artifact.mode
    labels = np.asarray(labels, dtype=int)
    if mode in {"manual_breadth2", "pseudo_breadth2"}:
        margin = np.abs(scored["breadth_score"] - artifact.thresholds["breadth_median"])
    elif mode in {"manual_stress2", "pseudo_stress2", "jump_stress2"}:
        margin = np.abs(scored["stress_score"] - artifact.thresholds["stress_median"])
    elif mode == "cluster_cov2":
        x = scored[SCORE_COLUMNS].values
        dist = ((x[:, None, :] - artifact.centroids[None, :, :]) ** 2).sum(axis=-1) ** 0.5
        sorted_dist = np.sort(dist, axis=1)
        margin = sorted_dist[:, 1] - sorted_dist[:, 0]
    elif mode == "pseudo_market3":
        score = scored["breadth_score"] - scored["stress_score"]
        low = artifact.thresholds["market_low"]
        high = artifact.thresholds["market_high"]
        margin = np.minimum(np.abs(score - low), np.abs(score - high))
    else:
        margin = np.zeros(len(scored), dtype=float)
    margin = np.asarray(margin, dtype=float)
    return np.clip(1.0 / (1.0 + np.exp(-margin)), 0.0, 1.0)


def _margin(scored, artifact):
    mode = artifact.mode
    if mode in {"manual_breadth2", "pseudo_breadth2"}:
        return scored["breadth_score"] - artifact.thresholds["breadth_median"]
    if mode in {"manual_stress2", "pseudo_stress2", "jump_stress2"}:
        return scored["stress_score"] - artifact.thresholds["stress_median"]
    if mode == "cluster_cov2":
        x = scored[SCORE_COLUMNS].values
        dist = ((x[:, None, :] - artifact.centroids[None, :, :]) ** 2).sum(axis=-1) ** 0.5
        return dist[:, 0] - dist[:, 1]
    if mode == "pseudo_market3":
        score = scored["breadth_score"] - scored["stress_score"]
        low = artifact.thresholds["market_low"]
        high = artifact.thresholds["market_high"]
        return np.minimum(score - low, high - score)
    return np.zeros(len(scored), dtype=float)


def apply_regime_artifact(feature_df, artifact):
    scored = _score_frame(feature_df.copy().reset_index(drop=True), artifact)
    labels = _assign_labels(scored, artifact)
    scored["regime_label"] = labels.astype(int)
    scored["regime_confidence"] = _confidence(scored, labels, artifact)
    scored["regime_margin"] = np.asarray(_margin(scored, artifact), dtype=float)
    return scored


def build_regime_table(price_data, mask_data, offsets_by_split, lookback, steps, mode, seed=0, jump_min_run=3):
    features = compute_regime_feature_table(price_data, mask_data, offsets_by_split, lookback, steps)
    artifact = fit_regime_artifact(features, mode, seed=seed, jump_min_run=jump_min_run)
    table = apply_regime_artifact(features, artifact)
    return table, artifact


def make_regime_lookup(regime_table):
    return {int(row["offset"]): row.to_dict() for _, row in regime_table.iterrows()}


def manual_gate_from_row(row, mode, num_regimes, temperature=0.5, eps=1e-6):
    if row is None or mode not in MANUAL_REGIME_MODES:
        return None
    num_regimes = int(num_regimes)
    label = int(row.get("regime_label", 0))
    if mode == "manual_breadth2":
        margin = float(row.get("regime_margin", row["breadth_score"])) / max(float(temperature), eps)
        p0 = 1.0 / (1.0 + np.exp(-margin))
        values = np.array([p0, 1.0 - p0], dtype=np.float32)
    elif mode == "manual_stress2":
        margin = float(row.get("regime_margin", row["stress_score"])) / max(float(temperature), eps)
        p1 = 1.0 / (1.0 + np.exp(-margin))
        values = np.array([1.0 - p1, p1], dtype=np.float32)
    else:
        values = np.full(num_regimes, 0.02 / max(num_regimes - 1, 1), dtype=np.float32)
        values[min(label, num_regimes - 1)] = 0.98
    if len(values) != num_regimes:
        padded = np.zeros(num_regimes, dtype=np.float32)
        padded[: min(len(values), num_regimes)] = values[:num_regimes]
        values = padded
    values = values / max(float(values.sum()), eps)
    return torch.from_numpy(values.astype(np.float32))


def gate_target_from_row(row, mode, device):
    if row is None or mode not in PSEUDO_REGIME_MODES:
        return None
    return torch.tensor([int(row.get("regime_label", 0))], dtype=torch.long, device=device)


def gate_summary(regime_table):
    out = {}
    if regime_table is None or regime_table.empty:
        return out
    for split, group in regime_table.groupby("split"):
        labels = group["regime_label"].astype(int).values
        counts = pd.Series(labels).value_counts(normalize=True).to_dict()
        switches = int(np.sum(labels[1:] != labels[:-1])) if len(labels) > 1 else 0
        out[f"{split}_regime_switch_count"] = switches
        out[f"{split}_regime_occupancy_min"] = float(min(counts.values())) if counts else 0.0
        out[f"{split}_regime_occupancy_max"] = float(max(counts.values())) if counts else 0.0
        for label, value in counts.items():
            out[f"{split}_regime_{int(label)}_occupancy"] = float(value)
    return out


def regime_table_to_json_records(regime_table):
    cols = ["split", "offset", "day_idx", *SCORE_COLUMNS, "regime_label", "regime_confidence", "regime_margin"]
    records = regime_table[cols].copy()
    return json.loads(records.to_json(orient="records"))
