from dataclasses import dataclass

import numpy as np
import torch

from model_rcls_delta import REGIME_FEATURE_NAMES


REGIME_MODES = (
    "legacy_delta",
    "pseudo_stress2",
    "jump_stress2",
    "manual_stress2",
    "pseudo_market3",
)
NEW_REGIME_MODES = set(REGIME_MODES) - {"legacy_delta"}
PSEUDO_REGIME_MODES = {"pseudo_stress2", "jump_stress2", "pseudo_market3"}
MANUAL_REGIME_MODES = {"manual_stress2"}
SCORE_COLUMNS = ("breadth_score", "stress_score", "corr_score")
REGIME_DIAGNOSTIC_COLUMNS = (
    "regime_mode",
    "regime_label",
    "regime_confidence",
    "regime_margin",
    "breadth_score",
    "stress_score",
    "corr_score",
)


@dataclass
class RegimeArtifact:
    mode: str
    num_regimes: int
    feature_mean: list
    feature_std: list
    thresholds: dict
    label_counts: dict
    jump_min_run: int = 3
    fit_split: str = "train"

    def to_metadata(self):
        return {
            "mode": self.mode,
            "num_regimes": int(self.num_regimes),
            "fit_split": self.fit_split,
            "feature_names": list(REGIME_FEATURE_NAMES),
            "feature_mean": [float(x) for x in self.feature_mean],
            "feature_std": [float(x) for x in self.feature_std],
            "thresholds": {str(k): float(v) for k, v in self.thresholds.items()},
            "label_counts": {str(k): int(v) for k, v in self.label_counts.items()},
            "jump_min_run": int(self.jump_min_run),
        }


def _safe_std(values):
    value = float(np.nanstd(values))
    return value if np.isfinite(value) and value > 1e-8 else 1.0


def _sigmoid(value):
    value = float(np.clip(value, -60.0, 60.0))
    return 1.0 / (1.0 + np.exp(-value))


def _feature_vector(row):
    return np.asarray([row.get(name, np.nan) for name in REGIME_FEATURE_NAMES], dtype=float)


def _zscore(values, artifact):
    values = np.asarray(values, dtype=float)
    mean = np.asarray(artifact.feature_mean, dtype=float)
    std = np.asarray(artifact.feature_std, dtype=float)
    return (values - mean) / std


def _scores_from_z(z):
    idx = {name: i for i, name in enumerate(REGIME_FEATURE_NAMES)}
    breadth = (
        z[idx["market_ret_mean"]]
        + 0.50 * z[idx["market_ret_last"]]
        + 0.75 * z[idx["frac_positive"]]
        - 0.50 * z[idx["downside_vol"]]
        - 0.25 * z[idx["market_ret_std"]]
    )
    stress = (
        z[idx["market_ret_std"]]
        + z[idx["downside_vol"]]
        + 0.50 * z[idx["max_abs_ret"]]
        + 0.50 * z[idx["synchronism"]]
        + 0.25 * z[idx["mean_abs_ret"]]
        - 0.25 * z[idx["market_ret_mean"]]
    )
    corr = (
        z[idx["synchronism"]]
        + 0.50 * z[idx["market_ret_std"]]
        - 0.50 * z[idx["dispersion"]]
    )
    return {
        "breadth_score": float(breadth),
        "stress_score": float(stress),
        "corr_score": float(corr),
    }


def score_feature_row(row, artifact):
    return _scores_from_z(_zscore(_feature_vector(row), artifact))


def _raw_label(scored_row, artifact):
    mode = artifact.mode
    if mode in {"pseudo_stress2", "jump_stress2", "manual_stress2"}:
        return int(scored_row["stress_score"] >= artifact.thresholds["stress_median"])
    if mode == "pseudo_market3":
        score = scored_row["breadth_score"] - scored_row["stress_score"]
        if score >= artifact.thresholds["market_high"]:
            return 0
        if score <= artifact.thresholds["market_low"]:
            return 2
        return 1
    return 0


def _online_min_run_smooth(labels, min_run):
    labels = [int(x) for x in labels]
    if not labels:
        return []
    current = labels[0]
    candidate = current
    count = 0
    output = []
    for label in labels:
        if label == current:
            candidate = current
            count = 0
        elif label == candidate:
            count += 1
        else:
            candidate = label
            count = 1
        if candidate != current and count >= int(min_run):
            current = candidate
            count = 0
        output.append(current)
    return output


def _confidence(scored_row, artifact):
    if artifact.mode in {"pseudo_stress2", "jump_stress2", "manual_stress2"}:
        margin = abs(scored_row["stress_score"] - artifact.thresholds["stress_median"])
    elif artifact.mode == "pseudo_market3":
        score = scored_row["breadth_score"] - scored_row["stress_score"]
        margin = min(
            abs(score - artifact.thresholds["market_low"]),
            abs(score - artifact.thresholds["market_high"]),
        )
    else:
        margin = 0.0
    return float(np.clip(_sigmoid(margin), 0.0, 1.0))


def _margin(scored_row, artifact):
    if artifact.mode in {"pseudo_stress2", "jump_stress2", "manual_stress2"}:
        return float(scored_row["stress_score"] - artifact.thresholds["stress_median"])
    if artifact.mode == "pseudo_market3":
        score = scored_row["breadth_score"] - scored_row["stress_score"]
        return float(
            min(
                score - artifact.thresholds["market_low"],
                artifact.thresholds["market_high"] - score,
            )
        )
    return 0.0


def fit_regime_artifact(feature_rows, mode, num_regimes, jump_min_run=3):
    if mode not in NEW_REGIME_MODES:
        raise ValueError("fit_regime_artifact only supports new regime modes")
    if mode == "pseudo_market3" and int(num_regimes) != 3:
        raise ValueError("pseudo_market3 requires num_regimes=3")
    if mode != "pseudo_market3" and int(num_regimes) != 2:
        raise ValueError("{} requires num_regimes=2".format(mode))

    train_rows = [row for row in feature_rows if row.get("split") == "train"]
    if not train_rows:
        raise ValueError("Cannot fit regime artifact without train rows")
    matrix = np.asarray([_feature_vector(row) for row in train_rows], dtype=float)
    mean = np.nanmean(matrix, axis=0)
    std = np.asarray([_safe_std(matrix[:, idx]) for idx in range(matrix.shape[1])])
    artifact = RegimeArtifact(
        mode=mode,
        num_regimes=int(num_regimes),
        feature_mean=mean.tolist(),
        feature_std=std.tolist(),
        thresholds={},
        label_counts={},
        jump_min_run=int(jump_min_run),
    )
    scored_train = [score_feature_row(row, artifact) for row in train_rows]
    stress_values = np.asarray([row["stress_score"] for row in scored_train], dtype=float)
    market_values = np.asarray(
        [row["breadth_score"] - row["stress_score"] for row in scored_train],
        dtype=float,
    )
    artifact.thresholds = {
        "stress_median": float(np.nanmedian(stress_values)),
        "market_low": float(np.nanquantile(market_values, 1.0 / 3.0)),
        "market_high": float(np.nanquantile(market_values, 2.0 / 3.0)),
    }
    applied = apply_regime_artifact(feature_rows, artifact)
    counts = {}
    for row in applied:
        if row.get("split") == "train":
            label = int(row["regime_label"])
            counts[label] = counts.get(label, 0) + 1
    artifact.label_counts = counts
    return artifact


def apply_regime_artifact(feature_rows, artifact):
    scored = []
    for row in feature_rows:
        out = dict(row)
        out.update(score_feature_row(row, artifact))
        out["_raw_label"] = _raw_label(out, artifact)
        scored.append(out)

    if artifact.mode == "jump_stress2":
        by_split = {}
        for idx, row in enumerate(scored):
            by_split.setdefault(row.get("split", ""), []).append(idx)
        labels = [int(row["_raw_label"]) for row in scored]
        for _, indices in by_split.items():
            indices = sorted(indices, key=lambda idx: int(scored[idx].get("offset", idx)))
            smoothed = _online_min_run_smooth(
                [labels[idx] for idx in indices],
                artifact.jump_min_run,
            )
            for idx, label in zip(indices, smoothed):
                labels[idx] = int(label)
    else:
        labels = [int(row["_raw_label"]) for row in scored]

    output = []
    for row, label in zip(scored, labels):
        row = dict(row)
        row.pop("_raw_label", None)
        row["regime_label"] = int(label)
        row["regime_confidence"] = _confidence(row, artifact)
        row["regime_margin"] = _margin(row, artifact)
        output.append(row)
    return output


def make_regime_lookup(regime_rows):
    return {int(row["offset"]): row for row in regime_rows}


def manual_gate_from_row(row, mode, num_regimes, temperature=0.5, eps=1e-6):
    if row is None or mode not in MANUAL_REGIME_MODES:
        return None
    num_regimes = int(num_regimes)
    if num_regimes != 2:
        raise ValueError("manual_stress2 requires num_regimes=2")
    margin = float(row.get("regime_margin", row.get("stress_score", 0.0)))
    p1 = _sigmoid(margin / max(float(temperature), eps))
    values = np.asarray([1.0 - p1, p1], dtype=np.float32)
    values = values / max(float(values.sum()), eps)
    return torch.from_numpy(values)


def switch_count(labels):
    values = [int(x) for x in labels]
    if len(values) <= 1:
        return 0
    return int(sum(1 for prev, cur in zip(values[:-1], values[1:]) if prev != cur))


def regime_summary(regime_rows):
    summary = {}
    by_split = {}
    for row in regime_rows:
        by_split.setdefault(row.get("split", ""), []).append(row)
    for split, rows in by_split.items():
        rows = sorted(rows, key=lambda row: int(row.get("offset", 0)))
        labels = [int(row["regime_label"]) for row in rows]
        counts = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        total = max(1, len(labels))
        shares = [count / total for count in counts.values()]
        summary["{}_regime_switch_count".format(split)] = switch_count(labels)
        summary["{}_regime_occupancy_min".format(split)] = min(shares) if shares else 0.0
        summary["{}_regime_occupancy_max".format(split)] = max(shares) if shares else 0.0
        for label, count in counts.items():
            summary["{}_regime_{}_occupancy".format(split, label)] = count / total
    return summary

