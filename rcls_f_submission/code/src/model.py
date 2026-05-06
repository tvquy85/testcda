import torch
import torch.nn as nn
import torch.nn.functional as F

from model_rcls import RCLSStockMixing


def build_activation(name):
    activations = {
        "gelu": nn.GELU,
        "hardswish": nn.Hardswish,
        "relu": nn.ReLU,
    }
    try:
        return activations[name.lower()]()
    except KeyError as exc:
        raise ValueError("Unsupported activation: {}".format(name)) from exc

def get_loss(prediction, ground_truth, base_price, mask, batch_size, alpha):
    device = prediction.device
    all_one = torch.ones(batch_size, 1, dtype=torch.float32).to(device)
    return_ratio = torch.div(torch.sub(prediction, base_price), base_price)
    reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
    pre_pw_dif = torch.sub(
        return_ratio @ all_one.t(),
        all_one @ return_ratio.t()
    )
    gt_pw_dif = torch.sub(
        all_one @ ground_truth.t(),
        ground_truth @ all_one.t()
    )
    mask_pw = mask @ mask.t()
    rank_loss = torch.mean(
        F.relu(pre_pw_dif * gt_pw_dif * mask_pw)
    )
    loss = reg_loss + alpha * rank_loss
    return loss, reg_loss, rank_loss, return_ratio


class MixerBlock(nn.Module):
    def __init__(self, mlp_dim, hidden_dim, dropout=0.0, activation="relu"):
        super(MixerBlock, self).__init__()
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.dense_1 = nn.Linear(mlp_dim, hidden_dim)
        self.LN = build_activation(activation)
        self.dense_2 = nn.Linear(hidden_dim, mlp_dim)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.LN(x)
        if self.dropout != 0.0:
            x = F.dropout(x, p=self.dropout)
        x = self.dense_2(x)
        if self.dropout != 0.0:
            x = F.dropout(x, p=self.dropout)
        return x


class Mixer2d(nn.Module):
    def __init__(self, time_steps, channels, activation="relu"):
        super(Mixer2d, self).__init__()
        self.LN_1 = nn.LayerNorm([time_steps, channels])
        self.LN_2 = nn.LayerNorm([time_steps, channels])
        self.timeMixer = MixerBlock(time_steps, time_steps, activation=activation)
        self.channelMixer = MixerBlock(channels, channels, activation=activation)

    def forward(self, inputs):
        x = self.LN_1(inputs)
        x = x.permute(0, 2, 1)
        x = self.timeMixer(x)
        x = x.permute(0, 2, 1)

        x = self.LN_2(x + inputs)
        y = self.channelMixer(x)
        return x + y


class TriU(nn.Module):
    def __init__(self, time_step):
        super(TriU, self).__init__()
        self.time_step = time_step
        self.triU = nn.ParameterList(
            [
                nn.Linear(i + 1, 1)
                for i in range(time_step)
            ]
        )

    def forward(self, inputs):
        x = self.triU[0](inputs[:, :, 0].unsqueeze(-1))
        for i in range(1, self.time_step):
            x = torch.cat([x, self.triU[i](inputs[:, :, 0:i + 1])], dim=-1)
        return x


class TimeMixerBlock(nn.Module):
    def __init__(self, time_step, activation="relu"):
        super(TimeMixerBlock, self).__init__()
        self.time_step = time_step
        self.dense_1 = TriU(time_step)
        self.LN = build_activation(activation)
        self.dense_2 = TriU(time_step)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.LN(x)
        x = self.dense_2(x)
        return x


class MultiScaleTimeMixer(nn.Module):
    def __init__(self, time_step, channel, scale_count=1, activation="relu"):
        super(MultiScaleTimeMixer, self).__init__()
        self.time_step = time_step
        self.scale_count = scale_count
        self.mix_layer = nn.ParameterList([nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=2 ** i, stride=2 ** i),
            TriU(int(time_step / 2 ** i)),
            build_activation(activation),
            TriU(int(time_step / 2 ** i))
        ) for i in range(scale_count)])
        self.mix_layer[0] = nn.Sequential(
            nn.LayerNorm([time_step, channel]),
            TriU(int(time_step)),
            build_activation(activation),
            TriU(int(time_step))
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.mix_layer[0](x)
        for i in range(1, self.scale_count):
            y = torch.cat((y, self.mix_layer[i](x)), dim=-1)
        return y


class Mixer2dTriU(nn.Module):
    def __init__(self, time_steps, channels, activation="relu"):
        super(Mixer2dTriU, self).__init__()
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
        super(MultTime2dMixer, self).__init__()
        self.mix_layer = Mixer2dTriU(time_step, channel, activation=main_activation)
        self.scale_mix_layer = Mixer2dTriU(scale_dim, channel, activation=scale_activation)

    def forward(self, inputs, y):
        y = self.scale_mix_layer(y)
        x = self.mix_layer(inputs)
        return torch.cat([inputs, x, y], dim=1)


class NoGraphMixer(nn.Module):
    def __init__(self, stocks, hidden_dim=20, activation="hardswish"):
        super(NoGraphMixer, self).__init__()
        self.dense1 = nn.Linear(stocks, hidden_dim)
        self.activation = build_activation(activation)
        self.dense2 = nn.Linear(hidden_dim, stocks)
        self.layer_norm_stock = nn.LayerNorm(stocks)

    def forward(self, inputs):
        x = inputs
        x = x.permute(1, 0)
        x = self.layer_norm_stock(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        x = x.permute(1, 0)
        return x


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
        model_name="stockmixer",
        gate_hidden=128,
        rcls_dropout=0.10,
    ):
        super(StockMixer, self).__init__()
        scale_dim = 8
        model_name = model_name.lower()
        if main_mixer_activation is None:
            main_mixer_activation = activation
        if scale_mixer_activation is None:
            scale_mixer_activation = activation
        if stock_activation is None:
            stock_activation = "hardswish"
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
        self.model_name = model_name
        self.last_regime_prob = None
        if model_name == "stockmixer":
            self.stock_mixer = NoGraphMixer(stocks, market, activation=stock_activation)
        elif model_name in ("rcls_f", "rcls_f_k3", "rcls_f_k1"):
            num_regimes = 1 if model_name == "rcls_f_k1" else 3
            self.stock_mixer = RCLSStockMixing(
                n_stocks=stocks,
                d_model=time_steps * 2 + scale_dim,
                market_dim=market,
                num_regimes=num_regimes,
                dropout=rcls_dropout,
                activation=stock_activation,
                gate_hidden=gate_hidden,
                return_gate=True,
            )
        else:
            raise ValueError("Unsupported model_name: {}".format(model_name))
        self.time_fc_ = nn.Linear(time_steps * 2 + scale_dim, 1)

    def forward(self, inputs):
        x = inputs.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        y = self.mixer(inputs, x)
        y = self.channel_fc(y).squeeze(-1)

        mixed = self.stock_mixer(y)
        if isinstance(mixed, tuple):
            z, pi = mixed
            self.last_regime_prob = pi.detach()
        else:
            z = mixed
            self.last_regime_prob = None
        y = self.time_fc(y)
        z = self.time_fc_(z)
        return y + z

