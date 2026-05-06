from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_activation(name):
    table = {"gelu": nn.GELU, "hardswish": nn.Hardswish, "relu": nn.ReLU}
    try:
        return table[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc


def get_price_loss(prediction, ground_truth, base_price, mask, batch_size, alpha_rank):
    device = prediction.device
    all_one = torch.ones(batch_size, 1, dtype=torch.float32, device=device)
    return_ratio = (prediction - base_price) / (base_price + 1e-12)
    point_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
    pre_pw_dif = return_ratio @ all_one.t() - all_one @ return_ratio.t()
    gt_pw_dif = all_one @ ground_truth.t() - ground_truth @ all_one.t()
    mask_pw = mask @ mask.t()
    rank_loss = torch.mean(F.relu(pre_pw_dif * gt_pw_dif * mask_pw))
    return point_loss + alpha_rank * rank_loss, point_loss, rank_loss, return_ratio


class MixerBlock(nn.Module):
    def __init__(self, mlp_dim, hidden_dim, activation="relu"):
        super().__init__()
        self.dense_1 = nn.Linear(mlp_dim, hidden_dim)
        self.activation = build_activation(activation)
        self.dense_2 = nn.Linear(hidden_dim, mlp_dim)

    def forward(self, x):
        return self.dense_2(self.activation(self.dense_1(x)))


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
        x = self.timeMixer(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.LN_2(x + inputs)
        return x + self.channelMixer(x)


class MultTime2dMixer(nn.Module):
    def __init__(self, time_step, channel, scale_dim=8, main_activation="relu", scale_activation="relu"):
        super().__init__()
        self.mix_layer = Mixer2dTriU(time_step, channel, activation=main_activation)
        self.scale_mix_layer = Mixer2dTriU(scale_dim, channel, activation=scale_activation)

    def forward(self, inputs, y):
        return torch.cat([inputs, self.mix_layer(inputs), self.scale_mix_layer(y)], dim=1)


class NoGraphMixer(nn.Module):
    def __init__(self, stocks, hidden_dim=20, activation="hardswish"):
        super().__init__()
        self.dense1 = nn.Linear(stocks, hidden_dim)
        self.activation = build_activation(activation)
        self.dense2 = nn.Linear(hidden_dim, stocks)
        self.layer_norm_stock = nn.LayerNorm(stocks)

    def forward(self, inputs, context=None):
        x = self.layer_norm_stock(inputs.permute(1, 0))
        x = self.dense2(self.activation(self.dense1(x)))
        aux = self._empty_aux(inputs.device, inputs.dtype)
        return x.permute(1, 0), aux

    @staticmethod
    def _empty_aux(device, dtype):
        pi = torch.ones(1, dtype=dtype, device=device)
        return {"pi": pi, "gate_entropy": torch.zeros((), dtype=dtype, device=device), "dominant_regime": torch.zeros((), dtype=torch.long, device=device)}


class ContextGatingUnit(nn.Module):
    def __init__(self, hidden_dim, context_dim):
        super().__init__()
        mid = max(hidden_dim // 2, 1)
        self.context_proj = nn.Sequential(nn.Linear(context_dim, mid), nn.Hardswish(), nn.Linear(mid, hidden_dim))

    def forward(self, x, context):
        trunk, gate = torch.chunk(x, chunks=2, dim=-1)
        if context.dim() == 1:
            context = context.unsqueeze(0)
        bias = self.context_proj(context.to(device=x.device, dtype=x.dtype))
        return F.hardswish(trunk) * torch.sigmoid(gate + bias)


class ContextGMLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, context_dim):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.channel_proj1 = nn.Linear(input_dim, hidden_dim * 2)
        self.gate = ContextGatingUnit(hidden_dim, context_dim)
        self.channel_proj2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, context):
        return x + self.channel_proj2(self.gate(self.channel_proj1(self.norm(x)), context))


class ContextStockMixer(nn.Module):
    def __init__(self, stocks, hidden_dim=20, context_dim=5, depth=3):
        super().__init__()
        self.blocks = nn.ModuleList([ContextGMLPBlock(stocks, hidden_dim, context_dim) for _ in range(depth)])

    def forward(self, inputs, context):
        x = inputs.permute(1, 0)
        for block in self.blocks:
            x = block(x, context)
        aux = NoGraphMixer._empty_aux(inputs.device, inputs.dtype)
        return x.permute(1, 0), aux


@dataclass
class RCLSCAGVariant:
    num_regimes: int
    uniform_gate: bool
    use_context: bool
    use_gate: bool
    cag_compatible: bool


def parse_rcls_cag_variant(model_name):
    if model_name == "stockmixer":
        return RCLSCAGVariant(1, True, False, False, False)
    if model_name in {"cag_mlp", "rcls_cag_k1"}:
        return RCLSCAGVariant(1, True, True, True, True)
    if model_name == "rcls_cag_k2":
        return RCLSCAGVariant(2, False, True, True, False)
    if model_name == "rcls_cag_k2_uniform":
        return RCLSCAGVariant(2, True, True, True, False)
    if model_name == "rcls_cag_k2_nocontext":
        return RCLSCAGVariant(2, False, False, True, False)
    if model_name == "rcls_cag_k2_nogate":
        return RCLSCAGVariant(2, False, True, False, False)
    if model_name == "rcls_cag_k3":
        return RCLSCAGVariant(3, False, True, True, False)
    if model_name == "rcls_cag_k3_uniform":
        return RCLSCAGVariant(3, True, True, True, False)
    raise ValueError(f"Unknown model: {model_name}")


class RCLSCAGMixer(nn.Module):
    def __init__(
        self,
        stocks,
        feature_dim,
        hidden_dim,
        context_dim,
        num_regimes=2,
        router_hidden=64,
        temperature=0.7,
        uniform_gate=False,
        use_context=True,
        use_gate=True,
        dropout=0.0,
        delta_scale=1.0,
    ):
        super().__init__()
        self.num_regimes = num_regimes
        self.temperature = temperature
        self.uniform_gate = uniform_gate
        self.use_context = use_context
        self.use_gate = use_gate
        self.delta_scale = delta_scale
        self.norm = nn.LayerNorm(stocks)
        self.proj1 = nn.ModuleList([nn.Linear(stocks, hidden_dim * 2) for _ in range(num_regimes)])
        self.proj2 = nn.ModuleList([nn.Linear(hidden_dim, stocks) for _ in range(num_regimes)])
        self.context_mlp = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(context_dim, max(hidden_dim // 2, 1)), nn.Hardswish(), nn.Linear(max(hidden_dim // 2, 1), hidden_dim))
                for _ in range(num_regimes)
            ]
        )
        router_in = context_dim + 4 * feature_dim
        self.router = nn.Sequential(nn.Linear(router_in, router_hidden), nn.Hardswish(), nn.Linear(router_hidden, num_regimes))
        self.dropout = nn.Dropout(dropout)

    def summarize_h(self, h):
        return torch.cat([h.mean(dim=0), h.std(dim=0, unbiased=False), h.max(dim=0).values, h.min(dim=0).values], dim=0)

    def route(self, h, context):
        if self.uniform_gate:
            return torch.full((self.num_regimes,), 1.0 / self.num_regimes, dtype=h.dtype, device=h.device)
        if context is None or not self.use_context:
            context = torch.zeros(self.context_mlp[0][0].in_features, dtype=h.dtype, device=h.device)
        z = torch.cat([context.to(h.device, h.dtype), self.summarize_h(h)], dim=0)
        return torch.softmax(self.router(z) / max(self.temperature, 1e-6), dim=-1)

    def forward(self, inputs, context):
        x = self.norm(inputs.permute(1, 0))
        pi = self.route(inputs, context)
        outs = []
        for idx in range(self.num_regimes):
            y = self.proj1[idx](x)
            trunk, gate = torch.chunk(y, chunks=2, dim=-1)
            trunk = F.hardswish(trunk)
            if self.use_gate:
                if context is None or not self.use_context:
                    bias = torch.zeros_like(trunk)
                else:
                    bias = self.context_mlp[idx](context.to(x.device, x.dtype)).unsqueeze(0)
                z = trunk * torch.sigmoid(gate + bias)
            else:
                z = trunk
            outs.append(self.proj2[idx](z).permute(1, 0))
        stacked = torch.stack(outs, dim=0)
        delta = torch.sum(pi.view(-1, 1, 1) * stacked, dim=0)
        out = inputs + self.dropout(self.delta_scale * delta)
        entropy = -(pi * torch.log(pi + 1e-12)).sum()
        aux = {
            "pi": pi,
            "gate_entropy": entropy,
            "dominant_regime": torch.argmax(pi),
            "expert_norm": torch.stack([o.norm() for o in outs]).mean(),
        }
        return out, aux

    def diversity_loss(self):
        if self.num_regimes < 2:
            return next(self.parameters()).new_tensor(0.0)
        mats = [F.normalize(layer.weight.flatten(), dim=0) for layer in self.proj1]
        loss = next(self.parameters()).new_tensor(0.0)
        count = 0
        for i in range(len(mats)):
            for j in range(i + 1, len(mats)):
                loss = loss + (mats[i] * mats[j]).sum().pow(2)
                count += 1
        return loss / max(count, 1)


class RCLSForecastModel(nn.Module):
    def __init__(
        self,
        model_name,
        stocks,
        time_steps,
        channels,
        market_dim,
        scale,
        context_dim,
        activation="hardswish",
        main_mixer_activation="hardswish",
        scale_mixer_activation="gelu",
        stock_activation="hardswish",
        router_hidden=64,
        router_temperature=0.7,
        num_layers=3,
    ):
        super().__init__()
        self.model_name = model_name
        self.variant = parse_rcls_cag_variant(model_name)
        scale_dim = 8
        feature_dim = time_steps * 2 + scale_dim
        self.mixer = MultTime2dMixer(time_steps, channels, scale_dim, main_mixer_activation, scale_mixer_activation)
        self.channel_fc = nn.Linear(channels, 1)
        self.time_fc = nn.Linear(feature_dim, 1)
        self.conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=2, stride=2)
        self.time_fc_ = nn.Linear(feature_dim, 1)
        if model_name == "stockmixer":
            self.stock_mixer = NoGraphMixer(stocks, market_dim, activation=stock_activation)
        elif self.variant.cag_compatible:
            self.stock_mixer = ContextStockMixer(stocks, market_dim, context_dim=context_dim, depth=num_layers)
        else:
            self.stock_mixer = RCLSCAGMixer(
                stocks=stocks,
                feature_dim=feature_dim,
                hidden_dim=market_dim,
                context_dim=context_dim,
                num_regimes=self.variant.num_regimes,
                router_hidden=router_hidden,
                temperature=router_temperature,
                uniform_gate=self.variant.uniform_gate,
                use_context=self.variant.use_context,
                use_gate=self.variant.use_gate,
            )

    def encode(self, inputs):
        x = self.conv(inputs.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.mixer(inputs, x)
        return self.channel_fc(y).squeeze(-1)

    def forward(self, inputs, context=None):
        h = self.encode(inputs)
        mixed, aux = self.stock_mixer(h, context)
        pred = self.time_fc(h) + self.time_fc_(mixed)
        aux["mu"] = pred
        return pred, aux

    def auxiliary_diversity_loss(self):
        if hasattr(self.stock_mixer, "diversity_loss"):
            return self.stock_mixer.diversity_loss()
        return next(self.parameters()).new_tensor(0.0)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
