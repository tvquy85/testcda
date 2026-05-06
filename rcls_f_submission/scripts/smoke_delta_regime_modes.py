import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "code" / "src"
sys.path.insert(0, str(SRC))

from model_rcls_delta import REGIME_FEATURE_NAMES  # noqa: E402
from regime_modes import (  # noqa: E402
    apply_regime_artifact,
    fit_regime_artifact,
    manual_gate_from_row,
    switch_count,
)


def feature_row(split, offset, profile):
    if profile == "risk_on":
        values = {
            "market_ret_mean": 0.025,
            "market_ret_std": 0.010,
            "market_ret_last": 0.030,
            "downside_vol": 0.001,
            "dispersion": 0.018,
            "synchronism": 0.45,
            "mean_abs_ret": 0.012,
            "max_abs_ret": 0.025,
            "frac_positive": 0.72,
        }
    elif profile == "neutral":
        values = {
            "market_ret_mean": 0.000,
            "market_ret_std": 0.020,
            "market_ret_last": 0.000,
            "downside_vol": 0.010,
            "dispersion": 0.025,
            "synchronism": 0.55,
            "mean_abs_ret": 0.020,
            "max_abs_ret": 0.045,
            "frac_positive": 0.50,
        }
    else:
        values = {
            "market_ret_mean": -0.025,
            "market_ret_std": 0.055,
            "market_ret_last": -0.030,
            "downside_vol": 0.045,
            "dispersion": 0.016,
            "synchronism": 0.82,
            "mean_abs_ret": 0.042,
            "max_abs_ret": 0.090,
            "frac_positive": 0.28,
        }
    return {"split": split, "offset": offset, "day_idx": offset + 16, **values}


def synthetic_rows(valid_extreme=False):
    rows = []
    profiles = ["risk_on", "neutral", "stress"]
    for i in range(36):
        rows.append(feature_row("train", i, profiles[i % len(profiles)]))
    for i in range(12):
        profile = "stress" if valid_extreme else profiles[(i + 1) % len(profiles)]
        rows.append(feature_row("valid", 100 + i, profile))
    for i in range(12):
        rows.append(feature_row("test", 200 + i, profiles[(i + 2) % len(profiles)]))
    return rows


def labels_for_split(rows, split):
    return [int(row["regime_label"]) for row in rows if row["split"] == split]


def assert_close(a, b, name):
    if not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("{} changed despite identical train split: {} vs {}".format(name, a, b))


def main():
    rows = synthetic_rows(valid_extreme=False)
    rows_with_extreme_valid = synthetic_rows(valid_extreme=True)

    stress = fit_regime_artifact(rows, "manual_stress2", num_regimes=2, jump_min_run=3)
    stress_extreme = fit_regime_artifact(rows_with_extreme_valid, "manual_stress2", num_regimes=2, jump_min_run=3)
    assert_close(stress.thresholds["stress_median"], stress_extreme.thresholds["stress_median"], "stress_median")

    applied_stress = apply_regime_artifact(rows, stress)
    high_stress = max(applied_stress, key=lambda row: row["stress_score"])
    if int(high_stress["regime_label"]) != 1:
        raise AssertionError("manual_stress2 must map high stress to regime 1")
    pi = manual_gate_from_row(high_stress, "manual_stress2", 2, temperature=0.5)
    if pi is None or float(pi[1]) <= float(pi[0]):
        raise AssertionError("manual_stress2 manual gate must favor regime 1")

    raw = fit_regime_artifact(rows, "pseudo_stress2", num_regimes=2, jump_min_run=3)
    jump = fit_regime_artifact(rows, "jump_stress2", num_regimes=2, jump_min_run=3)
    raw_labels = labels_for_split(apply_regime_artifact(rows, raw), "train")
    jump_labels = labels_for_split(apply_regime_artifact(rows, jump), "train")
    if switch_count(jump_labels) > switch_count(raw_labels):
        raise AssertionError("jump_stress2 must not increase train label switch count")

    market3 = fit_regime_artifact(rows, "pseudo_market3", num_regimes=3, jump_min_run=3)
    labels = set(labels_for_split(apply_regime_artifact(rows, market3), "train"))
    if labels != {0, 1, 2}:
        raise AssertionError("pseudo_market3 expected labels {0,1,2}, got {}".format(sorted(labels)))

    missing = [name for name in REGIME_FEATURE_NAMES if name not in rows[0]]
    if missing:
        raise AssertionError("synthetic regime rows missing features: {}".format(missing))

    print("RCLS-Delta regime mode smoke passed.")
    print("raw_switch_count={}".format(switch_count(raw_labels)))
    print("jump_switch_count={}".format(switch_count(jump_labels)))
    print("pseudo_market3_labels={}".format(sorted(labels)))


if __name__ == "__main__":
    main()

