from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_rcls_cag import RCLSForecastModel  # noqa: E402


def check_model(name):
    torch.manual_seed(0)
    n, t, f = 32, 16, 5
    context_dim = 5
    x = torch.randn(n, t, f)
    x[:, :, -1] = x[:, :, -1].abs() + 1.0
    ctx = torch.rand(context_dim)
    model = RCLSForecastModel(
        model_name=name,
        stocks=n,
        time_steps=t,
        channels=f,
        market_dim=8,
        scale=3,
        context_dim=context_dim,
        main_mixer_activation="hardswish",
        scale_mixer_activation="gelu",
        stock_activation="hardswish",
    )
    pred, aux = model(x, ctx)
    if pred.shape != (n, 1):
        raise AssertionError(f"{name} pred shape {pred.shape}")
    pi = aux["pi"]
    if abs(float(pi.sum()) - 1.0) > 1e-5:
        raise AssertionError(f"{name} pi does not sum to 1: {pi}")
    if "uniform" in name and pi.numel() > 1:
        expected = torch.full_like(pi, 1.0 / pi.numel())
        if not torch.allclose(pi, expected, atol=1e-6):
            raise AssertionError(f"{name} uniform pi mismatch: {pi}")
    if torch.isnan(pred).any():
        raise AssertionError(f"{name} produced NaN")


def main():
    for name in [
        "stockmixer",
        "cag_mlp",
        "rcls_cag_k1",
        "rcls_cag_k2",
        "rcls_cag_k2_uniform",
        "rcls_cag_k2_nocontext",
        "rcls_cag_k2_nogate",
        "rcls_cag_k3",
    ]:
        check_model(name)
    print("forward smoke passed")


if __name__ == "__main__":
    main()
