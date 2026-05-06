import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "code" / "src"
sys.path.insert(0, str(SRC))

import train  # noqa: E402


CASES = {
    "stockmixer": {
        "regime_mode": "legacy_delta",
        "gate_pseudo_label": False,
        "gate_pseudo_weight": 0.0,
        "gate_pseudo_final_weight": 0.0,
        "gate_confidence_weight": 0.0,
        "expert_diversity_weight": 0.0,
    },
    "rcls_delta_identity": {
        "regime_mode": "legacy_delta",
        "num_regimes": 1,
        "uniform_gate": True,
        "gate_pseudo_label": False,
        "gate_pseudo_weight": 0.0,
        "gate_pseudo_final_weight": 0.0,
        "gate_confidence_weight": 0.0,
        "expert_diversity_weight": 0.0,
        "delta_scale": 0.0,
        "delta_trainable_scale": False,
    },
    "rcls_delta_k1": {
        "regime_mode": "legacy_delta",
        "num_regimes": 1,
        "uniform_gate": False,
        "gate_pseudo_label": False,
        "gate_confidence_weight": 0.0,
        "expert_diversity_weight": 0.0,
    },
    "rcls_delta_k2": {
        "regime_mode": "legacy_delta",
        "num_regimes": 2,
        "uniform_gate": False,
        "gate_pseudo_label": True,
        "gate_pseudo_weight": 0.02,
        "gate_pseudo_final_weight": 0.005,
        "gate_confidence_weight": 0.0005,
        "expert_diversity_weight": 0.00001,
    },
    "rcls_delta_k2_uniform": {
        "regime_mode": "legacy_delta",
        "num_regimes": 2,
        "uniform_gate": True,
        "gate_pseudo_label": False,
        "gate_pseudo_weight": 0.0,
        "gate_pseudo_final_weight": 0.0,
        "gate_confidence_weight": 0.0,
        "expert_diversity_weight": 0.0,
    },
    "rcls_delta_k2_nostress": {
        "regime_mode": "legacy_delta",
        "num_regimes": 2,
        "uniform_gate": False,
        "gate_feature_mode": "embedding_only",
        "gate_pseudo_label": True,
    },
    "rcls_delta_k3": {
        "regime_mode": "legacy_delta",
        "num_regimes": 3,
        "uniform_gate": False,
        "gate_pseudo_label": True,
    },
    "rcls_delta_k3_uniform": {
        "regime_mode": "legacy_delta",
        "num_regimes": 3,
        "uniform_gate": True,
        "gate_pseudo_label": False,
        "gate_confidence_weight": 0.0,
        "expert_diversity_weight": 0.0,
    },
}

REGIME_CASES = {
    "rcls_delta_k2_pseudo_stress2": {
        "argv": ["--model", "rcls_delta_k2", "--regime-mode", "pseudo_stress2"],
        "expected": {
            "num_regimes": 2,
            "regime_mode": "pseudo_stress2",
            "gate_pseudo_label": True,
        },
    },
    "rcls_delta_k2_jump_stress2": {
        "argv": ["--model", "rcls_delta_k2", "--regime-mode", "jump_stress2"],
        "expected": {
            "num_regimes": 2,
            "regime_mode": "jump_stress2",
            "gate_pseudo_label": True,
        },
    },
    "rcls_delta_k2_manual_stress2": {
        "argv": ["--model", "rcls_delta_k2", "--regime-mode", "manual_stress2"],
        "expected": {
            "num_regimes": 2,
            "regime_mode": "manual_stress2",
            "gate_pseudo_label": False,
            "gate_pseudo_weight": 0.0,
            "gate_pseudo_final_weight": 0.0,
            "gate_confidence_weight": 0.0,
        },
    },
    "rcls_delta_k3_pseudo_market3": {
        "argv": ["--model", "rcls_delta_k3", "--regime-mode", "pseudo_market3"],
        "expected": {
            "num_regimes": 3,
            "regime_mode": "pseudo_market3",
            "gate_pseudo_label": True,
        },
    },
}


def assert_close(actual, expected, model, field):
    if isinstance(expected, float):
        if not math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-12):
            raise AssertionError(
                "{}.{} expected {!r}, got {!r}".format(model, field, expected, actual)
            )
        return
    if actual != expected:
        raise AssertionError(
            "{}.{} expected {!r}, got {!r}".format(model, field, expected, actual)
        )


def main():
    observed = {}
    for model, expected in CASES.items():
        args = train.parse_args(
            [
                "--dataset",
                "NASDAQ",
                "--model",
                model,
                "--epochs",
                "1",
                "--dry-run",
                "true",
            ]
        )
        config = train.selected_config(args)
        observed[model] = {key: config.get(key) for key in expected}
        for field, expected_value in expected.items():
            assert_close(config.get(field), expected_value, model, field)

    for name, case in REGIME_CASES.items():
        args = train.parse_args(
            [
                "--dataset",
                "NASDAQ",
                "--epochs",
                "1",
                "--dry-run",
                "true",
            ]
            + case["argv"]
        )
        config = train.selected_config(args)
        observed[name] = {key: config.get(key) for key in case["expected"]}
        for field, expected_value in case["expected"].items():
            assert_close(config.get(field), expected_value, name, field)

    print(json.dumps(observed, indent=2, sort_keys=True))
    print("RCLS-Delta config check passed.")


if __name__ == "__main__":
    main()
