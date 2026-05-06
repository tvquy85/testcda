# gmlp_core.py
"""
gmlp_core.py
=================

This module contains generic, reusable components for gated MLPs (gMLPs).  It
provides two flavours of gating units: one without any external context and
another which conditions the gates on a low‑dimensional context vector.  These
building blocks are assembled into gMLP blocks and full gMLP networks.  A
utility method is also supplied to compute simple gate‑saturation statistics to
aid in debugging and ablation studies.

Classes
-------

``GatingUnitNoContext``
    A gating unit that splits its input channels, applies an activation to
    the trunk and a sigmoid gate to the remainder.  No external context is
    used.

``GatingUnitWithContext``
    Similar to ``GatingUnitNoContext`` but accepts a small context vector
    which is projected to the hidden dimension and added to the gate branch
    before the sigmoid.  This allows the gating pattern to depend on
    high‑level market statistics.

``gMLPBlockNoContext`` and ``gMLPBlockWithContext``
    Implements a single residual block of a gMLP.  A layer normalisation
    precedes a linear projection, gating unit and a second linear projection.
    Dropout can optionally be applied after each linear layer for regularisation.

``gMLPNoContext`` and ``gMLPWithContext``
    Stacks multiple gMLP blocks into a complete network.  During the forward
    pass, each block updates the input sequentially.  A ``gate_saturation``
    method is provided to report what proportion of gating values lie near
    zero or one.

The functions/classes defined here are independent of any specific stock
mixing logic; they operate on generic tensors and can be re‑used in a wide
variety of settings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingUnitNoContext(nn.Module):
    """A gating unit without external context.

    Splits the last dimension of the input tensor into a trunk and a gate
    branch, applies a Hardswish activation to the trunk and a Sigmoid to the
    gate branch, then combines them element‑wise.  No external context is
    involved.

    Parameters
    ----------
    hidden_dim : int
        The number of channels in the hidden representation.  The input to
        this unit is expected to have size ``hidden_dim * 2`` along the last
        dimension.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.act_trunk = nn.Hardswish()
        self.act_gate = nn.Sigmoid()
        # Storage for the most recent gate values to allow gate saturation
        # statistics to be computed after a forward pass.
        self.last_gate: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (*, hidden_dim * 2)
        # Split trunk and gate
        u, v = torch.chunk(x, 2, dim=-1)
        u_act = self.act_trunk(u)
        g = self.act_gate(v)
        # Record gate values for later inspection (detach to avoid backprop)
        self.last_gate = g.detach()
        return u_act * g


class GatingUnitWithContext(nn.Module):
    """A gating unit that conditions gates on an external context vector.

    The design mirrors ``GatingUnitNoContext`` but adds a small feedforward
    network to project a context vector into the hidden dimension.  The
    projected context is added to the gate branch before applying the Sigmoid.

    Parameters
    ----------
    hidden_dim : int
        The number of channels in the hidden representation.  The input is
        expected to have size ``hidden_dim * 2`` along its last dimension.
    ctx_dim : int, default 5
        Dimensionality of the context vector provided during forward passes.
    """

    def __init__(self, hidden_dim: int, ctx_dim: int = 5) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ctx_dim = ctx_dim
        # Simple two‑layer MLP to expand context into the hidden dimension
        mid_dim = max(1, hidden_dim // 2)
        self.context_mlp = nn.Sequential(
            nn.Linear(ctx_dim, mid_dim),
            nn.Hardswish(),
            nn.Linear(mid_dim, hidden_dim)
        )
        self.act_trunk = nn.Hardswish()
        self.act_gate = nn.Sigmoid()
        self.last_gate: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # x shape: (*, hidden_dim * 2)
        # ctx shape: (ctx_dim,) or (batch, ctx_dim)
        u, v = torch.chunk(x, 2, dim=-1)
        u_act = self.act_trunk(u)
        # Project context and broadcast to match v
        ctx_proj = self.context_mlp(ctx)
        # If ctx has more than one dimension, collapse batch dims to 1 for addition
        # v shape: (*, hidden_dim)
        if ctx_proj.dim() == 1:
            # shape (hidden_dim,) -> (1, hidden_dim) for broadcasting
            ctx_proj = ctx_proj.unsqueeze(0)
        # Expand ctx_proj along the appropriate dimensions so that it can be
        # added to v.  We only match the last dimension of v.
        # If v.ndim == 2 (seq_len, hidden_dim), ctx_proj should be (1, hidden_dim)
        # If v.ndim == 3 (batch, seq_len, hidden_dim), ctx_proj should be
        # (batch, 1, hidden_dim).  Using unsqueeze(-2) handles both cases.
        ctx_expand = ctx_proj.unsqueeze(-2) if v.dim() >= 2 else ctx_proj
        g = self.act_gate(v + ctx_expand)
        self.last_gate = g.detach()
        return u_act * g


class gMLPBlockNoContext(nn.Module):
    """A single gMLP block without context.

    Performs layer normalisation, linear projection to a hidden dimension,
    element‑wise gating via ``GatingUnitNoContext``, followed by a second
    projection and residual connection.  Optional dropout can be applied
    after the linear projections.

    Parameters
    ----------
    input_dim : int
        The number of input channels (equal to the number of stocks when used
        for stock mixing).
    hidden_dim : int
        Size of the intermediate representation and gate hidden dimension.
    dropout_rate : float, default 0.0
        Dropout probability applied after each linear projection.
    layer_norm_eps : float, default 1e-5
        Epsilon for layer normalisation.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout_rate: float = 0.0, layer_norm_eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim, eps=layer_norm_eps)
        self.channel_proj1 = nn.Linear(input_dim, hidden_dim * 2)
        self.gate = GatingUnitNoContext(hidden_dim)
        self.channel_proj2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (seq_len, input_dim)
        residual = x
        x_norm = self.norm(x)
        x_proj = self.channel_proj1(x_norm)
        if self.dropout is not None:
            x_proj = self.dropout(x_proj)
        gated = self.gate(x_proj)
        out = self.channel_proj2(gated)
        if self.dropout is not None:
            out = self.dropout(out)
        return out + residual


class gMLPBlockWithContext(nn.Module):
    """A single gMLP block conditioned on a context vector.

    This block mirrors ``gMLPBlockNoContext`` but uses a context aware gating
    unit.  The same dropout configuration applies.
    """

    def __init__(self, input_dim: int, hidden_dim: int, ctx_dim: int = 5, dropout_rate: float = 0.0, layer_norm_eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim, eps=layer_norm_eps)
        self.channel_proj1 = nn.Linear(input_dim, hidden_dim * 2)
        self.gate = GatingUnitWithContext(hidden_dim, ctx_dim)
        self.channel_proj2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        x_proj = self.channel_proj1(x_norm)
        if self.dropout is not None:
            x_proj = self.dropout(x_proj)
        gated = self.gate(x_proj, ctx)
        out = self.channel_proj2(gated)
        if self.dropout is not None:
            out = self.dropout(out)
        return out + residual


class gMLPNoContext(nn.Module):
    """Stacks multiple context‑free gMLP blocks.

    Parameters
    ----------
    seq_len : int
        Unused in this implementation but provided for compatibility with
        context aware versions and future extensions.  A gMLP treats the
        sequence dimension independently of the gating logic.
    input_dim : int
        The dimensionality of the input (number of stocks).
    hidden_dim : int
        The hidden dimension for each block.
    depth : int
        Number of gMLP blocks to stack.
    dropout_rate : float, default 0.0
        Dropout probability shared across all blocks.
    layer_norm_eps : float, default 1e-5
        Epsilon for layer normalisation.
    """

    def __init__(self, seq_len: int, input_dim: int, hidden_dim: int = 128, depth: int = 2, dropout_rate: float = 0.0, layer_norm_eps: float = 1e-5) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.blocks = nn.ModuleList([
            gMLPBlockNoContext(input_dim=input_dim, hidden_dim=hidden_dim, dropout_rate=dropout_rate, layer_norm_eps=layer_norm_eps)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (seq_len, input_dim)
        for block in self.blocks:
            x = block(x)
        return x

    def gate_saturation(self) -> tuple[float, float]:
        """Compute gate saturation statistics across all blocks.

        Returns
        -------
        ratio_low : float
            The proportion of gating values strictly below 0.05 across all
            gating units and sequence positions.  If no gates have been
            computed yet (e.g. before the first forward pass), returns 0.
        ratio_high : float
            The proportion of gating values strictly above 0.95 across all
            gating units and sequence positions.
        """
        all_gates: list[torch.Tensor] = []
        for block in self.blocks:
            unit = block.gate
            if hasattr(unit, 'last_gate') and unit.last_gate is not None:
                all_gates.append(unit.last_gate.view(-1))
        if not all_gates:
            return 0.0, 0.0
        concat = torch.cat(all_gates)
        total = concat.numel()
        if total == 0:
            return 0.0, 0.0
        num_low = (concat < 0.05).sum().item()
        num_high = (concat > 0.95).sum().item()
        return num_low / total, num_high / total


class gMLPWithContext(nn.Module):
    """Stacks multiple context aware gMLP blocks.

    All blocks share the same context dimension.  The context vector should
    generally be of shape ``(ctx_dim,)`` when processing a single batch or
    ``(batch_size, ctx_dim)`` when processing multiple windows at once.  It is
    broadcast across the sequence dimension during gating.

    Parameters
    ----------
    seq_len : int
        Unused by the implementation but kept for API symmetry.
    input_dim : int
        The dimensionality of the input (number of stocks).
    hidden_dim : int
        The hidden dimension used inside each block.
    depth : int
        The number of gMLP blocks stacked sequentially.
    ctx_dim : int, default 5
        Dimensionality of the external context vector.
    dropout_rate : float, default 0.0
        Dropout probability applied after each linear projection.
    layer_norm_eps : float, default 1e-5
        Epsilon for layer normalisation.
    """

    def __init__(self, seq_len: int, input_dim: int, hidden_dim: int = 128, depth: int = 2, ctx_dim: int = 5, dropout_rate: float = 0.0, layer_norm_eps: float = 1e-5) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.ctx_dim = ctx_dim
        self.blocks = nn.ModuleList([
            gMLPBlockWithContext(input_dim=input_dim, hidden_dim=hidden_dim, ctx_dim=ctx_dim, dropout_rate=dropout_rate, layer_norm_eps=layer_norm_eps)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # x shape: (seq_len, input_dim)
        for block in self.blocks:
            x = block(x, ctx)
        return x

    def gate_saturation(self) -> tuple[float, float]:
        # Same logic as in gMLPNoContext
        all_gates: list[torch.Tensor] = []
        for block in self.blocks:
            unit = block.gate
            if hasattr(unit, 'last_gate') and unit.last_gate is not None:
                all_gates.append(unit.last_gate.view(-1))
        if not all_gates:
            return 0.0, 0.0
        concat = torch.cat(all_gates)
        total = concat.numel()
        if total == 0:
            return 0.0, 0.0
        num_low = (concat < 0.05).sum().item()
        num_high = (concat > 0.95).sum().item()
        return num_low / total, num_high / total
