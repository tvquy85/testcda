from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rcls_metrics import pearson_ic, rank_ic  # noqa: E402


def assert_close(name, value, expected, tol=1e-6):
    if not np.isfinite(value) or abs(value - expected) > tol:
        raise AssertionError(f"{name}: got {value}, expected {expected}")


def main():
    target = np.arange(100, dtype=float)
    assert_close("ic_identity", pearson_ic(target, target), 1.0)
    assert_close("rankic_identity", rank_ic(target, target), 1.0)
    assert_close("ic_negative", pearson_ic(-target, target), -1.0)
    assert_close("rankic_negative", rank_ic(-target, target), -1.0)
    assert_close("rankic_monotonic", rank_ic(target ** 3, target), 1.0)
    rng = np.random.default_rng(0)
    random_ic = pearson_ic(rng.normal(size=10000), rng.normal(size=10000))
    if abs(random_ic) > 0.05:
        raise AssertionError(f"random IC too large: {random_ic}")
    print("metric checks passed")


if __name__ == "__main__":
    main()
