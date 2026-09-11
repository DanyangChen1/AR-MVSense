from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class ConvEncoder(nn.Module):
    def __init__(self, dim: int, lightweight: bool = False) -> None:
        super().__init__()
        # The paper omits its channel/stride table.  The lightweight branch uses
        # a narrower stem and one extra spatial reduction so its compute follows
        # the reported FLOP ratio instead of only matching the parameter count.
        mid = max(32, 3 * dim // 8) if lightweight else max(24, dim // 2)
        self.stem = nn.Sequential(
            nn.Conv3d(3, mid, (3, 7, 7), stride=(1, 4, 4), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(mid),
            nn.GELU(),
        )
        if lightweight:
            self.refine = nn.Sequential(
                # RePhys Fig. 1 preserves T through its stages; reduce only H/W.
                nn.Conv3d(mid, mid, 3, stride=(1, 2, 2), padding=1, groups=mid, bias=False),
                nn.BatchNorm3d(mid),
                nn.GELU(),
                nn.Conv3d(mid, dim, 1, bias=False),
                nn.BatchNorm3d(dim),
                nn.GELU(),
            )
        else:
            self.refine = nn.Sequential(
                nn.Conv3d(mid, dim, 3, stride=(2, 1, 1), padding=1, bias=False),
                nn.BatchNorm3d(dim),
                nn.GELU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.stem(x))


class CrossScaleAttentionBlock(nn.Module):
    """Cross-attention from pooled high-resolution features to a lower scale."""

    def __init__(self, dim: int, heads: int, spatial_tokens: int, dropout: float) -> None:
        super().__init__()
        self.spatial_tokens = spatial_tokens
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, time, height, width = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
        high = F.adaptive_avg_pool2d(frames, (self.spatial_tokens, self.spatial_tokens))
        low = F.avg_pool2d(frames, 2, ceil_mode=True)
        low = F.adaptive_avg_pool2d(low, (self.spatial_tokens, self.spatial_tokens))
        q = high.flatten(2).transpose(1, 2)
        kv = low.flatten(2).transpose(1, 2)
        attended, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv), need_weights=False)
        tokens = attended + self.ffn(self.norm_ffn(attended))
        delta = tokens.transpose(1, 2).reshape(
            batch * time, channels, self.spatial_tokens, self.spatial_tokens
        )
        delta = F.interpolate(delta, size=(height, width), mode="bilinear", align_corners=False)
        delta = delta.reshape(batch, time, channels, height, width).permute(0, 2, 1, 3, 4)
        return x + self.gamma * delta


class Figure2CrossScaleAttentionBlock(nn.Module):
    """CCA -> Add&Norm -> S-T feed-forward -> Add&Norm, matching main-paper Fig. 2."""

    def __init__(self, dim: int, heads: int, spatial_tokens: int, dropout: float) -> None:
        super().__init__()
        self.spatial_tokens = spatial_tokens
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.norm1 = ChannelLayerNorm5d(dim)
        self.ffn = SpatialTemporalFeedForward(dim, dropout)
        self.norm2 = ChannelLayerNorm5d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, time, height, width = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
        high = F.adaptive_avg_pool2d(frames, (self.spatial_tokens, self.spatial_tokens))
        low = F.avg_pool2d(frames, 2, ceil_mode=True)
        low = F.adaptive_avg_pool2d(low, (self.spatial_tokens, self.spatial_tokens))
        q = high.flatten(2).transpose(1, 2)
        kv = low.flatten(2).transpose(1, 2)
        attended, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv), need_weights=False)
        delta = attended.transpose(1, 2).reshape(
            batch * time, channels, self.spatial_tokens, self.spatial_tokens
        )
        delta = F.interpolate(delta, size=(height, width), mode="bilinear", align_corners=False)
        delta = delta.reshape(batch, time, channels, height, width).permute(0, 2, 1, 3, 4)
        x = self.norm1(x + self.gamma * delta)
        return self.norm2(x + self.ffn(x))


class SlidingWindowTemporalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, window_size: int, shift: int, dropout: float) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def _window_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        left = self.shift
        right = (self.window_size - (time + left) % self.window_size) % self.window_size
        padded = F.pad(x, (0, 0, left, right))
        valid = F.pad(torch.ones((batch, time), dtype=torch.bool, device=x.device), (left, right))
        windows = padded.reshape(batch, -1, self.window_size, channels).flatten(0, 1)
        masks = ~valid.reshape(batch, -1, self.window_size).flatten(0, 1)
        attended, _ = self.attn(windows, windows, windows, key_padding_mask=masks, need_weights=False)
        restored = attended.reshape(batch, -1, channels)
        return restored[:, left : left + time]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._window_attention(self.norm1(x))
        return x + self.ffn(self.norm2(x))


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    kernel = conv.weight
    bias = conv.bias if conv.bias is not None else torch.zeros(kernel.size(0), device=kernel.device)
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return kernel * scale.reshape(-1, 1, 1, 1), bn.bias + (bias - bn.running_mean) * scale


class RepConv2d(nn.Module):
    """MobileOne-style depthwise branches followed by one pointwise projection."""

    def __init__(self, channels: int, deploy: bool = False) -> None:
        super().__init__()
        self.channels = channels
        self.deploy = deploy
        self.activation = nn.GELU()
        if deploy:
            self.reparam = nn.Conv2d(
                channels, channels, 3, padding=1, groups=channels, bias=True
            )
            self.pointwise_reparam = nn.Conv2d(channels, channels, 1, bias=True)
        else:
            self.branch_3x3 = nn.Sequential(
                nn.Conv2d(
                    channels, channels, 3, padding=1, groups=channels, bias=False
                ),
                nn.BatchNorm2d(channels),
            )
            self.branch_1x1 = nn.Sequential(
                nn.Conv2d(channels, channels, 1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
            )
            self.branch_identity = nn.BatchNorm2d(channels)
            self.pointwise = nn.Sequential(
                nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.activation(self.pointwise_reparam(self.reparam(x)))
        branches = self.branch_3x3(x) + self.branch_1x1(x) + self.branch_identity(x)
        return self.activation(self.pointwise(branches))

    def equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.deploy:
            return self.reparam.weight, self.reparam.bias
        k3, b3 = _fuse_conv_bn(self.branch_3x3[0], self.branch_3x3[1])
        k1, b1 = _fuse_conv_bn(self.branch_1x1[0], self.branch_1x1[1])
        k1 = F.pad(k1, (1, 1, 1, 1))
        identity = torch.zeros_like(k3)
        identity[:, 0, 1, 1] = 1.0
        identity_conv = nn.Conv2d(
            self.channels,
            self.channels,
            3,
            padding=1,
            groups=self.channels,
            bias=False,
        ).to(k3.device)
        identity_conv.weight.data.copy_(identity)
        kid, bid = _fuse_conv_bn(identity_conv, self.branch_identity)
        return k3 + k1 + kid, b3 + b1 + bid

    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        kernel, bias = self.equivalent_kernel_bias()
        reparam = nn.Conv2d(
            self.channels,
            self.channels,
            3,
            padding=1,
            groups=self.channels,
            bias=True,
        ).to(kernel.device)
        reparam.weight.data.copy_(kernel)
        reparam.bias.data.copy_(bias)
        self.reparam = reparam
        pointwise_kernel, pointwise_bias = _fuse_conv_bn(self.pointwise[0], self.pointwise[1])
        pointwise = nn.Conv2d(self.channels, self.channels, 1, bias=True).to(kernel.device)
        pointwise.weight.data.copy_(pointwise_kernel)
        pointwise.bias.data.copy_(pointwise_bias)
        self.pointwise_reparam = pointwise
        del self.branch_3x3
        del self.branch_1x1
        del self.branch_identity
        del self.pointwise
        self.deploy = True


class PartialSpatialConv(nn.Module):
    def __init__(self, dim: int, ratio: float) -> None:
        super().__init__()
        self.selected = max(1, int(round(dim * ratio)))
        self.conv = nn.Conv2d(self.selected, self.selected, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        selected, skipped = x[:, : self.selected], x[:, self.selected :]
        return torch.cat((self.conv(selected), skipped), dim=1)


class SqueezeExcitation(nn.Module):
    def __init__(self, dim: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, dim // reduction)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.net(x.mean(dim=(2, 3, 4)))
        return x * scale[:, :, None, None, None]


class ChannelLayerNorm5d(nn.Module):
    """LayerNorm over channels while preserving B,C,T,H,W layout."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)


class SpatialTemporalFeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The architecture hands spatially aggregated features to temporal
        # modeling; applying the channel MLP once per time step avoids silently
        # turning a temporal block into a per-pixel quadratic-cost operator.
        pooled = x.mean(dim=(3, 4)).transpose(1, 2)
        return self.net(pooled).transpose(1, 2)[:, :, :, None, None]


class ReparameterizedCCABlock(nn.Module):
    """Fig. 3 re-parameterized CCA: multi-branch conv, PConv, SE and two residual norms."""

    def __init__(self, dim: int, pconv_ratio: float, dropout: float, deploy: bool) -> None:
        super().__init__()
        self.rep = RepConv2d(dim, deploy=deploy)
        self.pconv = PartialSpatialConv(dim, pconv_ratio)
        self.se = SqueezeExcitation(dim)
        self.norm1 = ChannelLayerNorm5d(dim)
        self.ffn = SpatialTemporalFeedForward(dim, dropout)
        self.norm2 = ChannelLayerNorm5d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, time, height, width = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
        frames = self.pconv(self.rep(frames))
        transformed = frames.reshape(batch, time, channels, height, width).permute(0, 2, 1, 3, 4)
        x = self.norm1(x + self.se(transformed))
        return self.norm2(x + self.ffn(x))

    def switch_to_deploy(self) -> None:
        self.rep.switch_to_deploy()


class TASABlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        # Fig. 3 explicitly derives the adaptive shift weight from a gating CNN.
        # Depthwise temporal gating keeps the operation linear in sequence length.
        self.gating_cnn = nn.Conv1d(dim, dim, 3, padding=1, groups=dim, bias=True)
        nn.init.zeros_(self.gating_cnn.weight)
        nn.init.zeros_(self.gating_cnn.bias)
        self.spatial_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.se = SqueezeExcitation(dim)
        self.norm1 = ChannelLayerNorm5d(dim)
        self.ffn = SpatialTemporalFeedForward(dim, dropout)
        self.norm2 = ChannelLayerNorm5d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous = torch.cat((x[:, :, :1], x[:, :, :-1]), dim=2)
        following = torch.cat((x[:, :, 1:], x[:, :, -1:]), dim=2)
        gate = torch.tanh(self.gating_cnn(x.mean(dim=(3, 4))))[:, :, :, None, None]
        shifted = x + gate * (previous - following)
        batch, channels, time, height, width = shifted.shape
        frames = shifted.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
        frames = self.spatial_conv(frames)
        spatial = frames.reshape(batch, time, channels, height, width).permute(0, 2, 1, 3, 4)
        x = self.norm1(x + self.se(spatial))
        return self.norm2(x + self.ffn(x))


class RPPGViT(nn.Module):
    def __init__(
        self,
        dim: int = 124,
        heads: int = 4,
        cca_depth: int = 3,
        swta_depth: int = 4,
        window_size: int = 16,
        spatial_tokens: int = 8,
        dropout: float = 0.3,
        cca_variant: str = "token_ffn",
    ) -> None:
        super().__init__()
        self.encoder = ConvEncoder(dim, lightweight=False)
        if cca_variant == "token_ffn":
            cca_type = CrossScaleAttentionBlock
        elif cca_variant == "figure2":
            cca_type = Figure2CrossScaleAttentionBlock
        else:
            raise ValueError(f"Unknown CCA variant: {cca_variant}")
        self.cca = nn.ModuleList(
            [cca_type(dim, heads, spatial_tokens, dropout) for _ in range(cca_depth)]
        )
        self.swta = nn.ModuleList(
            [
                SlidingWindowTemporalBlock(
                    dim, heads, window_size, 0 if index % 2 == 0 else window_size // 2, dropout
                )
                for index in range(swta_depth)
            ]
        )
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        output_length = video.shape[1]
        x = video.permute(0, 2, 1, 3, 4)
        x = self.encoder(x)
        for block in self.cca:
            x = block(x)
        temporal = x.mean(dim=(3, 4)).transpose(1, 2)
        for block in self.swta:
            temporal = block(temporal)
        waveform = self.head(temporal).squeeze(-1)
        return F.interpolate(waveform[:, None], size=output_length, mode="linear", align_corners=False).squeeze(1)


class LightRPPGViT(nn.Module):
    def __init__(
        self,
        dim: int = 137,
        rep_depth: int = 3,
        tasa_depth: int = 4,
        pconv_ratio: float = 0.25,
        dropout: float = 0.2,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = ConvEncoder(dim, lightweight=True)
        self.rep_blocks = nn.ModuleList(
            [
                ReparameterizedCCABlock(dim, pconv_ratio, dropout, deploy)
                for _ in range(rep_depth)
            ]
        )
        self.tasa = nn.ModuleList([TASABlock(dim, dropout) for _ in range(tasa_depth)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        output_length = video.shape[1]
        x = self.encoder(video.permute(0, 2, 1, 3, 4))
        for block in self.rep_blocks:
            x = block(x)
        for block in self.tasa:
            x = block(x)
        temporal = x.mean(dim=(3, 4)).transpose(1, 2)
        waveform = self.head(temporal).squeeze(-1)
        return F.interpolate(waveform[:, None], size=output_length, mode="linear", align_corners=False).squeeze(1)

    def switch_to_deploy(self) -> None:
        for block in self.rep_blocks:
            block.switch_to_deploy()

    def deployed_copy(self) -> "LightRPPGViT":
        model = copy.deepcopy(self).eval()
        model.switch_to_deploy()
        return model


def build_model(config: dict[str, Any]) -> nn.Module:
    name = config["name"].lower()
    kwargs = {key: value for key, value in config.items() if key != "name"}
    if name == "rppgvit":
        return RPPGViT(**kwargs)
    if name in {"light_rppgvit", "light-rppgvit"}:
        return LightRPPGViT(**kwargs)
    raise ValueError(f"Unknown model: {config['name']}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
