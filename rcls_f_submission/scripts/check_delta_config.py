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
        "gate_pseudo_label": False,
        "gate_pseudo_weight": 0.0,
        "gate_pseudo_final_weight": 0.0,
        "gate_confidence_weight": 0.0,
        "expert_diversity_weight": 0.0,
    },
    "rcls_delta_identity": {
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
        "num_regimes": 1,
        "uniform_gate": False,
        "gate_pseudo_label": False,
        "gate_confidence_weight": 0.0,
        "expert_diversity_weight": 0.0,
    },
    "rcls_delta_k2": {
        "num_regimes": 2,
        "uniform_gate": False,
        "gate_pseudo_label": True,
        "gate_pseudo_weight": 0.02,
        "gate_pseudo_final_weight": 0.005,
        "gate_confidence_weight": 0.0005,
        "expert_diversity_weight": 0.00001,
    },
    "rcls_delta_k2_uniform": {
        "num_regimes": 2,
        "uniform_gate": True,
        "gate_pseudo_label": False,
        "gate_pseudo_weight": 0.0,
        "gate_pseudo_final_weight": 0.0,
        "gate_confidence_weight": 0.0,
        "expert_diversity_weight": 0.0,
    },
    "rcls_delta_k2_nostress": {
        "num_regimes": 2,
        "uniform_gate": False,
        "gate_feature_mode": "embedding_only",
        "gate_pseudo_label": True,
    },
    "rcls_delta_k3": {
        "num_regimes": 3,
        "uniform_gate": False,
        "gate_pseudo_label": True,
    },
    "rcls_delta_k3_uniform": {
        "num_regimes": 3,
        "uniform_gate": True,
        "gate_pseudo_label": False,
        "gate_confidence_weight": 0.0,
        "expert_diversity_weight": 0.0,
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

    print(json.dumps(observed, indent=2, sort_keys=True))
    print("RCLS-Delta config check passed.")


if __name__ == "__main__":
    main()
