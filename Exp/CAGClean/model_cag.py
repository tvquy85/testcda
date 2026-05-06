from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_activation(name: str) -> nn.Module:
    activations = {
        "gelu": nn.GELU,
        "hardswish": nn.Hardswish,
        "relu": nn.ReLU,
    }
    try:
        return activations[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc


def get_loss(prediction, ground_truth, base_price, mask, batch_size, alpha):
    device = prediction.device
    all_one = torch.ones(batch_size, 1, dtype=torch.float32, device=device)
    return_ratio = torch.div(torch.sub(prediction, base_price), base_price)
    reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
    pre_pw_dif = torch.sub(return_ratio @ all_one.t(), all_one @ return_ratio.t())
    gt_pw_dif = torch.sub(all_one @ ground_truth.t(), ground_truth @ all_one.t())
    mask_pw = mask @ mask.t()
    rank_loss = torch.mean(F.relu(pre_pw_dif * gt_pw_dif * mask_pw))
    loss = reg_loss + alpha * rank_loss
    return loss, reg_loss, rank_loss, return_ratio


class MixerBlock(nn.Module):
    def __init__(self, mlp_dim, hidden_dim, dropout=0.0, activation="relu"):
        super().__init__()
        self.dropout = dropout
        self.dense_1 = nn.Linear(mlp_dim, hidden_dim)
        self.activation = build_activation(activation)
        self.dense_2 = nn.Linear(hidden_dim, mlp_dim)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        if self.dropout != 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.dense_2(x)
        if self.dropout != 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class TriU(nn.Module):
    def __init__(self, time_step):
        super().__init__()
        self.triU = nn.ParameterList([nn.Linear(i + 1, 1) for i in range(time_step)])

    def forward(self, inputs):
        x = self.triU[0](inputs[:, :, 0].unsqueeze(-1))
        for i in range(1, len(self.triU)):
            x = torch.cat([x, self.triU[i](inputs[:, :, 0 : i + 1])], dim=-1)
        return x


class Mixer2dTriU(nn.Module):
    def __init__(self, time_steps, channels, activation="relu"):
        super().__init__()
        self.LN_1 = nn.LayerNorm([time_steps, channels])
        self.LN_2 = nn.LayerNorm([time_steps, channels])
        self.timeMixer = TriU(time_steps)
        self.channelMixer = MixerBlock(channels, channels, activation=activation)

    def forward(self, inputs):
        x = self.LN_1(inputs)
        x = x.permute(0, 2, 1)
        x = self.timeMixer(x)
        x = x.permute(0, 2, 1)
        x = self.LN_2(x + inputs)
        y = self.channelMixer(x)
        return x + y


class MultTime2dMixer(nn.Module):
    def __init__(
        self,
        time_step,
        channel,
        scale_dim=8,
        main_activation="relu",
        scale_activation="relu",
    ):
        super().__init__()
        self.mix_layer = Mixer2dTriU(time_step, channel, activation=main_activation)
        self.scale_mix_layer = Mixer2dTriU(scale_dim, channel, activation=scale_activation)

    def forward(self, inputs, y):
        y = self.scale_mix_layer(y)
        x = self.mix_layer(inputs)
        return torch.cat([inputs, x, y], dim=1)


class NoGraphMixer(nn.Module):
    def __init__(self, stocks, hidden_dim=20, activation="hardswish"):
        super().__init__()
        self.dense1 = nn.Linear(stocks, hidden_dim)
        self.activation = build_activation(activation)
        self.dense2 = nn.Linear(hidden_dim, stocks)
        self.layer_norm_stock = nn.LayerNorm(stocks)

    def forward(self, inputs):
        x = inputs.permute(1, 0)
        x = self.layer_norm_stock(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        return x.permute(1, 0)


class ContextGatingUnit(nn.Module):
    def __init__(self, hidden_dim, context_dim=5):
        super().__init__()
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, max(hidden_dim // 2, 1)),
            nn.Hardswish(),
            nn.Linear(max(hidden_dim // 2, 1), hidden_dim),
        )
        self.trunk_activation = nn.Hardswish()
        self.gate_activation = nn.Sigmoid()

    def forward(self, x, context):
        trunk, gate = torch.chunk(x, chunks=2, dim=-1)
        if context.dim() == 1:
            context = context.unsqueeze(0)
        context_bias = self.context_proj(context.to(dtype=x.dtype, device=x.device))
        return self.trunk_activation(trunk) * self.gate_activation(gate + context_bias)


class ContextGMLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, context_dim=5, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.channel_proj1 = nn.Linear(input_dim, hidden_dim * 2)
        self.gating = ContextGatingUnit(hidden_dim, context_dim=context_dim)
        self.channel_proj2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        residual = x
        x = self.norm(x)
        x = self.channel_proj1(x)
        x = self.gating(x, context)
        x = self.channel_proj2(x)
        x = self.dropout(x)
        return x + residual


class ContextGMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=20, context_dim=5, depth=3, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ContextGMLPBlock(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    context_dim=context_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x, context):
        for block in self.blocks:
            x = block(x, context)
        return x


class ContextStockMixer(nn.Module):
    def __init__(self, stocks, hidden_dim=20, depth=3, context_dim=5, dropout=0.0):
        super().__init__()
        self.gmlp = ContextGMLP(
            input_dim=stocks,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            depth=depth,
            dropout=dropout,
        )

    def forward(self, inputs, context):
        x = inputs.permute(1, 0)
        x = self.gmlp(x, context)
        return x.permute(1, 0)


class StockMixer(nn.Module):
    def __init__(
        self,
        stocks,
        time_steps,
        channels,
        market,
        scale,
        activation="hardswish",
        main_mixer_activation=None,
        scale_mixer_activation=None,
        stock_activation="hardswish",
    ):
        super().__init__()
        scale_dim = 8
        if main_mixer_activation is None:
            main_mixer_activation = activation
        if scale_mixer_activation is None:
            scale_mixer_activation = activation
        self.mixer = MultTime2dMixer(
            time_steps,
            channels,
            scale_dim=scale_dim,
            main_activation=main_mixer_activation,
            scale_activation=scale_mixer_activation,
        )
        self.channel_fc = nn.Linear(channels, 1)
        self.time_fc = nn.Linear(time_steps * 2 + scale_dim, 1)
        self.conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=2, stride=2)
        self.stock_mixer = NoGraphMixer(stocks, market, activation=stock_activation)
        self.time_fc_ = nn.Linear(time_steps * 2 + scale_dim, 1)

    def forward(self, inputs, context=None):
        x = inputs.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        y = self.mixer(inputs, x)
        y = self.channel_fc(y).squeeze(-1)
        z = self.stock_mixer(y)
        y = self.time_fc(y)
        z = self.time_fc_(z)
        return y + z


class CAGStockMixer(nn.Module):
    def __init__(
        self,
        stocks,
        time_steps,
        channels,
        market,
        scale,
        activation="hardswish",
        main_mixer_activation=None,
        scale_mixer_activation=None,
        depth=3,
        context_dim=5,
        dropout=0.0,
    ):
        super().__init__()
        scale_dim = 8
        if main_mixer_activation is None:
            main_mixer_activation = activation
        if scale_mixer_activation is None:
            scale_mixer_activation = activation
        self.mixer = MultTime2dMixer(
            time_steps,
            channels,
            scale_dim=scale_dim,
            main_activation=main_mixer_activation,
            scale_activation=scale_mixer_activation,
        )
        self.channel_fc = nn.Linear(channels, 1)
        self.time_fc = nn.Linear(time_steps * 2 + scale_dim, 1)
        self.conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=2, stride=2)
        self.stock_mixer = ContextStockMixer(
            stocks,
            hidden_dim=market,
            depth=depth,
            context_dim=context_dim,
            dropout=dropout,
        )
        self.time_fc_ = nn.Linear(time_steps * 2 + scale_dim, 1)

    def forward(self, inputs, context):
        x = inputs.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        y = self.mixer(inputs, x)
        y = self.channel_fc(y).squeeze(-1)
        z = self.stock_mixer(y, context)
        y = self.time_fc(y)
        z = self.time_fc_(z)
        return y + z


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
