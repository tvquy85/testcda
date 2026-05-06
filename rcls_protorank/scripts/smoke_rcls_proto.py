import sys
from pathlib import Path

import torch
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from model_rcls_proto import RCLSProtoConfig, RCLSProtoRank  # noqa: E402
from regime import (  # noqa: E402
    apply_regime_artifact,
    fit_regime_artifact,
    manual_gate_from_row,
)
from train_rcls_proto import transform_rank_target_np  # noqa: E402


def assert_close(name, value, limit):
    if float(value) >= limit:
        raise AssertionError(f"{name}={float(value):.8g} must be < {limit}")


def run_forward(cfg):
    torch.manual_seed(7)
    x = torch.randn(128, cfg.lookback, cfg.n_features)
    x[:, :, -1] = torch.cumsum(torch.randn(128, cfg.lookback) * 0.01, dim=1) + 10.0
    mask = torch.ones(128, 1)
    model = RCLSProtoRank(cfg)
    model.eval()
    with torch.no_grad():
        return model(x, mask), x, mask, model


def test_shapes():
    cfg = RCLSProtoConfig(n_features=5, lookback=16, d_model=32, num_regimes=2, num_prototypes=8)
    out, _, _, _ = run_forward(cfg)
    assert out["score"].shape == (128,)
    assert out["mu"].shape == (128,)
    assert out["sigma"].shape == (128,)
    assert out["pi"].shape == (2,)
    assert torch.isfinite(out["score"]).all()
    assert torch.isfinite(out["sigma"]).all()
    cfg = RCLSProtoConfig(
        n_features=5,
        lookback=16,
        d_model=32,
        num_regimes=3,
        num_prototypes=8,
        architecture="v2",
        decouple_rank_score=True,
        use_stress_aux=True,
    )
    out, _, _, _ = run_forward(cfg)
    assert out["pi"].shape == (3,)
    assert out["rank_score"].shape == (128,)
    assert out["aux_stress_logits"].shape == (2,)
    if int(out["entity_pairwise_elements"]) != 0:
        raise AssertionError("v2 mixer must not allocate entity-by-entity attention")


def test_permutation_equivariance():
    cfg = RCLSProtoConfig(n_features=5, lookback=16, d_model=32, num_regimes=2, num_prototypes=8, dropout=0.0)
    out, x, mask, model = run_forward(cfg)
    perm = torch.randperm(x.shape[0])
    with torch.no_grad():
        out_perm = model(x[perm], mask[perm])
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(len(perm))
    err = (out_perm["score"][inv] - out["score"]).abs().max()
    assert_close("permutation_equivariance_max_error", err, 1e-4)
    cfg = RCLSProtoConfig(
        n_features=5,
        lookback=16,
        d_model=32,
        num_regimes=3,
        num_prototypes=8,
        dropout=0.0,
        architecture="v2",
        decouple_rank_score=True,
        use_stress_aux=True,
    )
    out, x, mask, model = run_forward(cfg)
    perm = torch.randperm(x.shape[0])
    with torch.no_grad():
        out_perm = model(x[perm], mask[perm])
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(len(perm))
    err = (out_perm["score"][inv] - out["score"]).abs().max()
    assert_close("v2_permutation_equivariance_max_error", err, 1e-4)


def test_variants():
    k1 = RCLSProtoConfig(n_features=5, lookback=16, d_model=32, num_regimes=1, num_prototypes=8)
    out, _, _, _ = run_forward(k1)
    if out["pi"].numel() != 1 or not torch.allclose(out["pi"], torch.ones_like(out["pi"])):
        raise AssertionError("k1 gate must be one-hot uniform over one regime")

    uniform = RCLSProtoConfig(
        n_features=5,
        lookback=16,
        d_model=32,
        num_regimes=2,
        num_prototypes=8,
        uniform_gate=True,
    )
    out, _, _, _ = run_forward(uniform)
    if not torch.allclose(out["pi"], torch.full_like(out["pi"], 0.5), atol=0.0, rtol=0.0):
        raise AssertionError("uniform_gate must be exactly uniform")

    static = RCLSProtoConfig(
        n_features=5,
        lookback=16,
        d_model=32,
        num_regimes=2,
        num_prototypes=8,
        static_proto=True,
    )
    out, _, _, _ = run_forward(static)
    if float(out["proto_delta_norm"]) != 0.0:
        raise AssertionError("static_proto must report proto_delta_norm=0")

    no_uncert = RCLSProtoConfig(
        n_features=5,
        lookback=16,
        d_model=32,
        num_regimes=2,
        num_prototypes=8,
        use_uncertainty=False,
    )
    out, _, _, _ = run_forward(no_uncert)
    if not torch.allclose(out["sigma"], torch.ones_like(out["sigma"])):
        raise AssertionError("no_uncert must report sigma=1")
    if not torch.allclose(out["score"], out["mu"]):
        raise AssertionError("no_uncert must report score=mu")


def make_feature_frame():
    rows = []
    for split, start, count in [("train", 0, 30), ("valid", 30, 8), ("test", 38, 8)]:
        for i in range(count):
            j = start + i
            pos = (j % 10) / 9.0
            stress = 1.0 - pos if j % 2 else pos
            rows.append(
                {
                    "split": split,
                    "offset": j,
                    "day_idx": j + 16,
                    "market_ret_mean": pos - 0.5,
                    "market_ret_std": 0.02 + 0.03 * stress,
                    "market_ret_last": pos - 0.5,
                    "downside_vol": 0.01 + 0.04 * stress,
                    "dispersion": 0.02 + 0.02 * stress,
                    "synchronism": 0.30 + 0.50 * stress,
                    "mean_abs_ret": 0.01 + 0.03 * stress,
                    "max_abs_ret": 0.02 + 0.05 * stress,
                    "frac_positive": 0.25 + 0.50 * pos,
                }
            )
    return pd.DataFrame(rows)


def switch_count(labels):
    labels = list(labels)
    return sum(int(a != b) for a, b in zip(labels[:-1], labels[1:]))


def test_regime_modes():
    features = make_feature_frame()
    breadth = fit_regime_artifact(features, "manual_breadth2", seed=0)
    applied = apply_regime_artifact(features, breadth)
    high = applied.sort_values("breadth_score", ascending=False).iloc[0]
    low = applied.sort_values("breadth_score", ascending=True).iloc[0]
    if int(high["regime_label"]) != 0 or int(low["regime_label"]) != 1:
        raise AssertionError("manual_breadth2 must map high breadth to risk-on label 0")
    pi = manual_gate_from_row(high.to_dict(), "manual_breadth2", 2, temperature=0.5)
    if pi is None or not (float(pi[0]) > float(pi[1])):
        raise AssertionError("manual_breadth2 gate must favor regime 0 for high breadth")

    stress = fit_regime_artifact(features, "manual_stress2", seed=0)
    applied = apply_regime_artifact(features, stress)
    high_stress = applied.sort_values("stress_score", ascending=False).iloc[0]
    if int(high_stress["regime_label"]) != 1:
        raise AssertionError("manual_stress2 must map high stress to label 1")

    raw = fit_regime_artifact(features, "pseudo_stress2", seed=0)
    jump = fit_regime_artifact(features, "jump_stress2", seed=0, jump_min_run=3)
    raw_labels = apply_regime_artifact(features, raw).sort_values("offset")["regime_label"]
    jump_labels = apply_regime_artifact(features, jump).sort_values("offset")["regime_label"]
    if switch_count(jump_labels) > switch_count(raw_labels):
        raise AssertionError("jump_stress2 must not increase switch count")

    cluster_a = fit_regime_artifact(features, "cluster_cov2", seed=0)
    cluster_b = fit_regime_artifact(features, "cluster_cov2", seed=0)
    if not (cluster_a.centroids == cluster_b.centroids).all():
        raise AssertionError("cluster_cov2 must be deterministic under fixed seed")

    for mode in ["pseudo_breadth2", "pseudo_stress2", "jump_stress2", "pseudo_market3"]:
        artifact = fit_regime_artifact(features, mode, seed=0)
        counts = artifact.label_counts
        min_frac = min(counts.values()) / sum(counts.values())
        if min_frac < 0.10:
            raise AssertionError(f"{mode} has an under-populated train pseudo-label")


def test_target_transforms():
    target = pd.Series([1.0, 2.0, 3.0]).values
    mask = pd.Series([1.0, 1.0, 1.0]).values
    residual = transform_rank_target_np(target, mask, "market_residual")
    if abs(float(residual.mean())) > 1e-8:
        raise AssertionError("market_residual target transform must be day-centered")
    z = transform_rank_target_np(target, mask, "day_zscore")
    if abs(float(z.mean())) > 1e-8 or abs(float(z.std()) - 1.0) > 1e-6:
        raise AssertionError("day_zscore target transform must be centered and unit scaled")
    raw = transform_rank_target_np(target, mask, "raw")
    if not (raw == target).all():
        raise AssertionError("raw target transform must be identity")


def main():
    test_shapes()
    test_permutation_equivariance()
    test_variants()
    test_regime_modes()
    test_target_transforms()
    print("RCLS-ProtoRank smoke checks passed.")


if __name__ == "__main__":
    main()
