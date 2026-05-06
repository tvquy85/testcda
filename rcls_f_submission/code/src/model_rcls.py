import torch
import torch.nn as nn


class RCLSStockMixing(nn.Module):
    """
    Regime-Conditioned Low-Rank Set Mixer.

    Supports [N, D] and [B, N, D] inputs. The local StockMixer integration uses
    [N, D], where N is the number of stocks and D is the post-temporal feature
    dimension before the final time projection.
    """

    def __init__(
        self,
        n_stocks,
        d_model,
        market_dim,
        num_regimes=3,
        dropout=0.10,
        activation="hardswish",
        gate_hidden=128,
        return_gate=True,
        uniform_gate=False,
    ):
        super(RCLSStockMixing, self).__init__()
        self.n_stocks = int(n_stocks)
        self.d_model = int(d_model)
        self.market_dim = int(market_dim)
        self.num_regimes = int(num_regimes)
        self.return_gate = bool(return_gate)
        self.uniform_gate = bool(uniform_gate)

        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)

        act = activation.lower()
        if act == "hardswish":
            self.act = nn.Hardswish()
        elif act == "relu":
            self.act = nn.ReLU()
        elif act == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError("Unsupported activation: {}".format(activation))

        self.M1 = nn.Parameter(
            torch.randn(self.num_regimes, self.market_dim, self.n_stocks) * 0.02
        )
        self.M2 = nn.Parameter(
            torch.randn(self.num_regimes, self.n_stocks, self.market_dim) * 0.02
        )
        self.gate = nn.Sequential(
            nn.Linear(4 * self.d_model, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, self.num_regimes),
        )
        self.last_regime_prob = None

    def _summary(self, z):
        mean = z.mean(dim=1)
        std = z.std(dim=1, unbiased=False)
        maxv = z.max(dim=1).values
        minv = z.min(dim=1).values
        return torch.cat([mean, std, maxv, minv], dim=-1)

    def forward(self, h):
        single = False
        if h.dim() == 2:
            h = h.unsqueeze(0)
            single = True
        if h.dim() != 3:
            raise ValueError(
                "RCLSStockMixing expects [N,D] or [B,N,D], got {}".format(
                    tuple(h.shape)
                )
            )

        batch_size, n_stocks, d_model = h.shape
        if n_stocks != self.n_stocks:
            raise ValueError(
                "Expected n_stocks={}, got N={}".format(self.n_stocks, n_stocks)
            )
        if d_model != self.d_model:
            raise ValueError(
                "Expected d_model={}, got D={}".format(self.d_model, d_model)
            )

        z = self.norm(h)
        if self.uniform_gate:
            pi = torch.full(
                (batch_size, self.num_regimes),
                1.0 / self.num_regimes,
                dtype=h.dtype,
                device=h.device,
            )
        else:
            pi = torch.softmax(self.gate(self._summary(z)), dim=-1)

        u = torch.einsum("kmn,bnd->bkmd", self.M1, z)
        u = self.act(u)
        o = torch.einsum("knm,bkmd->bknd", self.M2, u)
        o = (pi[:, :, None, None] * o).sum(dim=1)
        out = h + self.dropout(o)
        self.last_regime_prob = pi.detach()

        if single:
            out = out.squeeze(0)
            pi = pi.squeeze(0)

        if self.return_gate:
            return out, pi
        return out
