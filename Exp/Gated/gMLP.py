import torch
import torch.nn as nn


class GatingUnit(nn.Module):
    def __init__(self, dim, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ln = nn.Sequential(
            nn.Linear(5, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )
        self.acv_trunk = nn.Hardswish()
        self.acv_gate = nn.Sigmoid()

    def forward(self, x):
        # Split channels
        u, v = torch.chunk(x, chunks=2, dim=-1)

        u = self.acv_trunk(u)
        v = self.acv_gate(v)

        # Element-wise multiplication with u
        return u * v


class gMLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, dropout_rate=0.0):
        super().__init__()

        self.norm = nn.LayerNorm(input_dim)
        self.channel_proj1 = nn.Linear(input_dim, hidden_dim * 2)
        self.sgu = GatingUnit(hidden_dim, seq_len)
        self.channel_proj2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.channel_proj1(x)
        x = self.sgu(x)
        x = self.channel_proj2(x)
        return x + residual


class gMLP(nn.Module):
    def __init__(
            self,
            seq_len,
            input_dim,
            hidden_dim=128,
            depth=2,
            dropout_rate=0.1
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            gMLPBlock(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                seq_len=seq_len,
                dropout_rate=dropout_rate
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
