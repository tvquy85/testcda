import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "code" / "src"
sys.path.insert(0, str(SRC))

from model import StockMixer  # noqa: E402


N_STOCKS = 1026
T_STEPS = 16
N_FEATURES = 5
MARKET_DIM = 20


def build_model(model_name, seed=123, **kwargs):
    torch.manual_seed(seed)
    return StockMixer(
        stocks=N_STOCKS,
        time_steps=T_STEPS,
        channels=N_FEATURES,
        market=MARKET_DIM,
        scale=3,
        activation="hardswish",
        main_mixer_activation="hardswish",
        scale_mixer_activation="gelu",
        stock_activation="hardswish",
        model_name=model_name,
        **kwargs,
    )


def assert_shape(value, expected, name):
    if tuple(value.shape) != tuple(expected):
        raise AssertionError("{} expected shape {}, got {}".format(name, expected, tuple(value.shape)))


def main():
    torch.manual_seed(20260506)
    x = torch.randn(N_STOCKS, T_STEPS, N_FEATURES)

    stockmixer = build_model("stockmixer")
    identity = build_model(
        "rcls_delta_identity",
        num_regimes=1,
        delta_scale=0.0,
        delta_trainable_scale=False,
        uniform_gate=True,
    )
    stockmixer.eval()
    identity.eval()
    with torch.no_grad():
        stock_pred = stockmixer(x)
        identity_pred = identity(x)
    assert_shape(identity_pred, (N_STOCKS, 1), "identity prediction")
    max_diff = (stock_pred - identity_pred).abs().max().item()
    if max_diff > 1e-8:
        raise AssertionError("identity differs from stockmixer with shared seed: {}".format(max_diff))
    if float(identity.stock_mixer.delta_scale) != 0.0:
        raise AssertionError("identity delta_scale is not zero")

    uniform = build_model(
        "rcls_delta_k2_uniform",
        num_regimes=2,
        delta_scale=0.05,
        uniform_gate=True,
    )
    uniform.eval()
    with torch.no_grad():
        uniform_pred = uniform(x)
    assert_shape(uniform_pred, (N_STOCKS, 1), "uniform prediction")
    pi = uniform.last_regime_prob.detach().cpu()
    if pi.ndim == 2:
        pi = pi[0]
    if tuple(pi.shape) != (2,):
        raise AssertionError("uniform pi expected shape (2,), got {}".format(tuple(pi.shape)))
    if not torch.allclose(pi, torch.tensor([0.5, 0.5]), atol=1e-6, rtol=0.0):
        raise AssertionError("uniform pi is not exactly uniform: {}".format(pi.tolist()))

    learned = build_model("rcls_delta_k2", num_regimes=2, delta_scale=0.05, uniform_gate=False)
    learned.eval()
    with torch.no_grad():
        learned_pred = learned(x)
    assert_shape(learned_pred, (N_STOCKS, 1), "learned prediction")
    pi = learned.last_regime_prob.detach().cpu()
    if pi.ndim == 2:
        pi = pi[0]
    if tuple(pi.shape) != (2,):
        raise AssertionError("learned pi expected shape (2,), got {}".format(tuple(pi.shape)))
    if not torch.isclose(pi.sum(), torch.tensor(1.0), atol=1e-6, rtol=0.0):
        raise AssertionError("learned pi does not sum to 1: {}".format(float(pi.sum())))
    assert_shape(learned.stock_mixer.M1, (2, MARKET_DIM, N_STOCKS), "M1")
    assert_shape(learned.stock_mixer.M2, (2, N_STOCKS, MARKET_DIM), "M2")
    if learned.last_delta_norm is None or learned.last_base_norm is None:
        raise AssertionError("learned forward did not populate delta/base norms")

    print("RCLS-Delta forward smoke passed.")
    print("identity_max_abs_diff={:.3e}".format(max_diff))
    print("uniform_pi={}".format([float(v) for v in uniform.last_regime_prob.detach().cpu()]))
    print("learned_pi={}".format([float(v) for v in pi]))


if __name__ == "__main__":
    main()
