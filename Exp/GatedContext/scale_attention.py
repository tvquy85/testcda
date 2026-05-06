import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleAttention(nn.Module):
    def __init__(self, time_steps, channels, num_scales=3):
        super().__init__()
        self.scales = [time_steps // (2 ** i) for i in range(num_scales)]
        self.attention_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(channels, channels),
                nn.Softmax(dim=-1)
            ) for _ in range(num_scales)
        ])

    def forward(self, features):
        attended_features = []
        for i, scale in enumerate(self.scales):
            feature = features[i]
            attention = self.attention_blocks[i](feature)
            attended = attention * feature
            attended_features.append(attended)

        return attended_features


class MultiScaleGating(nn.Module):
    def __init__(self, time_steps, channels, num_scales=3):
        super().__init__()
        self.scales = [time_steps // (2 ** i) for i in range(num_scales)]
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(channels, channels),
                nn.Sigmoid()
            ) for _ in range(num_scales)
        ])

    def forward(self, features):
        gated_features = []
        for i, scale in enumerate(self.scales):
            # Reshape và áp dụng gating
            feature = features[i]
            gate = self.gates[i](feature)
            gated = gate * feature
            gated_features.append(gated)

        return gated_features


class MultiScaleTemporalConv(nn.Module):
    def __init__(self, time_steps, channels, num_scales=3):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=2 ** i, stride=2 ** i),
                nn.BatchNorm1d(channels),
                nn.ReLU()
            ) for i in range(num_scales)
        ])

    def forward(self, x):
        # x: [batch, time_steps, channels]
        x = x.permute(0, 2, 1)  # [batch, channels, time_steps]
        conv_features = []
        for conv in self.convs:
            conv_out = conv(x)
            conv_features.append(conv_out)

        # Upsample các features về cùng kích thước
        max_time = max(f.size(-1) for f in conv_features)
        upsampled_features = []
        for f in conv_features:
            upsampled = F.interpolate(f, size=max_time, mode='linear')
            upsampled_features.append(upsampled)

        return torch.cat(upsampled_features, dim=1)


class HierarchicalFeatureFusion(nn.Module):
    def __init__(self, time_steps, channels, num_scales=3):
        super().__init__()
        self.scales = [time_steps // (2 ** i) for i in range(num_scales)]
        self.convs = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size=1)
            for _ in range(num_scales)
        ])

        self.fusion_weights = nn.Parameter(torch.ones(num_scales) / num_scales)

    def forward(self, features):
        # features: list of tensors from different scales
        # Áp dụng softmax để chuẩn hóa weights
        weights = F.softmax(self.fusion_weights, dim=0)

        # Weighted sum of features
        fused = None
        for i, (weight, feature, conv) in enumerate(zip(weights, features, self.convs)):
            feature = feature.permute(0, 2, 1)
            feature = conv(feature)
            feature = feature.permute(0, 2, 1)
            if fused is None:
                fused = feature
            else:
                fused = F.interpolate(fused.permute(0, 2, 1), size=fused.size(1) * 2, mode='linear').permute(0, 2, 1)
                fused += feature

        return fused


class AdaptiveScaleSelector(nn.Module):
    def __init__(self, time_steps, channels, num_scales=3):
        super().__init__()
        self.scale_selector = nn.Sequential(
            nn.Linear(channels * num_scales, num_scales),
            nn.Softmax(dim=-1)
        )

    def forward(self, features):
        # features: list of tensors from different scales
        # Concatenate features along channel dimension
        concat_features = torch.cat(features, dim=-1)

        # Predict scale importance
        scale_weights = self.scale_selector(concat_features)

        # Weighted combination
        weighted_features = []
        for i, (weight, feature) in enumerate(zip(scale_weights, features)):
            weighted_features.append(weight.unsqueeze(-1) * feature)

        return torch.sum(torch.stack(weighted_features), dim=0)


class AdaptiveScaleMixer(nn.Module):
    def __init__(self, scales, in_channels, hidden_dim):
        # 1. Dynamic Weighting
        self.scales = scales
        self.scale_weights = nn.Parameter(torch.ones(len(scales)))
        self.softmax = nn.Softmax(dim=0)
        # 2. Scale-specific Processing
        self.scale_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim, kernel_size=3),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ) for _ in scales
        ])
        # 3. Adaptive Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(scales), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        # 1. Process each scale
        scale_features = []
        for i, scale in enumerate(self.scales):
            # Scale-specific processing
            feat = self.scale_processors[i](x[scale])
            scale_features.append(feat)
        # 2. Dynamic weighting
        weights = self.softmax(self.scale_weights)
        weighted_features = [w * f for w, f in zip(weights, scale_features)]
        # 3. Adaptive fusion
        fused = torch.cat(weighted_features, dim=1)
        output = self.fusion(fused)
