import math
from collections import OrderedDict

import torch
import torch.nn as nn
from einops import rearrange

# import drop path from folder layers
from Layers.helper import DropPath

from .attention import (
    ALiBiAttention,
    Attention,
    AxialAttention,
    ConvAttention,
    EMAttention,
    LinAngularAttention,
    LocalityFeedForward,
    MultiCHA,
    MultiDPHConvHeadAttention,
    RelativeAttention,
    ResidualAttention,
    RobustAttention,
    RoformerAttention,
)
from .mlp import Mlp, RobustMlp


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        drop=0.0,
        mlp_ratio=4.0,
        activation=nn.GELU,
        norm_layer=nn.LayerNorm,
    ) -> None:
        """
        Transformer encoder block.

        params:
            :dim: Dimensionality of each token
            :mlp_ratio: MLP hidden dimensionality multiplier
        """
        super(Block, self).__init__()

        self.attention = Attention(dim, dropout=drop, num_heads=num_heads)
        self.MLP = Mlp(
            dim, dropout=drop, activation_function=activation, mlp_ratio=mlp_ratio
        )
        self.LN_1 = norm_layer(dim)
        self.LN_2 = norm_layer(dim)
        self.drop = nn.Dropout(drop)
        self.dropPath = DropPath(drop_prob=drop)

    def forward(self, x, mask=None):
        x = x + self.dropPath(self.attention(self.LN_1(x), mask=mask))
        x = x + self.dropPath(self.MLP(self.LN_2(x)))
        return x


class Parallel_blocks(nn.Module):
    """Parallel ViT block (N parallel attention followed by N parallel MLP)
    Based on:
      `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795

    from
    1) x'_l+1 = x+ mhsal_l( x)       -  2) x_l+1= x'_l+1 + mlp_l(x'_l+1)
    3) x'_l+2 = x+ mhsal_l+1( x_l+2) -  4) x_l+2= x'_l+2 + mlp_l+1(x'_l+2)

    to
    1) x_l+1 = x     + mhsal_1(x)   + mhsal_2(x)
    2) x_l+2 = x_l+1 + mlp_1(x_l+1) + mlp_2(x_l+1)
    """

    def __init__(
        self,
        dim,
        num_heads,
        drop=0.0,
        mlp_ratio=4.0,
        activation=nn.GELU,
        num_parallel=2,
    ) -> None:
        """
        Transformer encoder block.

        params:
            :dim: Dimensionality of each token
            :num_heads: Number of attention heads
            :mlp_ratio: MLP hidden dimensionality multiplier
            :num_parallel: Number of parallel blocks
        """
        super(Parallel_blocks, self).__init__()
        self.num_parallel = num_parallel
        self.attns = nn.ModuleList()
        self.mlps = nn.ModuleList()

        for _ in range(num_parallel):
            self.attns.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("norm", nn.LayerNorm(dim)),
                            ("attn", Attention(dim, dropout=drop, num_heads=num_heads)),
                            (
                                "drop_path",
                                DropPath(drop) if drop > 0.0 else nn.Identity(),
                            ),
                        ]
                    )
                )
            )
            self.mlps.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("norm", nn.LayerNorm(dim)),
                            (
                                "mlp",
                                Mlp(
                                    dim,
                                    dropout=drop,
                                    mlp_ratio=mlp_ratio,
                                    activation_function=activation,
                                ),
                            ),
                            (
                                "drop_path",
                                DropPath(drop) if drop > 0.0 else nn.Identity(),
                            ),
                        ]
                    )
                )
            )

    def _forward_jit(self, x, mask):
        x = x + torch.stack([attn(x, mask=mask) for attn in self.attns]).sum(dim=0)
        x = x + torch.stack([mlp(x) for mlp in self.mlps]).sum(dim=0)
        return x

    @torch.jit.ignore
    def _forward(self, x, mask):
        x = x + sum(attn(x) for attn in self.attns)
        x = x + sum(mlp(x) for mlp in self.mlps)
        return x

    def forward(self, x, mask=None):
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return self._forward_jit(x, mask=mask)
        else:
            return self._forward(x, mask=mask)


class CustomBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        drop=0.0,
        mlp_ratio=4.0,
        activation=nn.GELU,
        attention=LinAngularAttention,
        mlp=Mlp,
    ) -> None:
        """
        Transformer encoder block with a custom Attention layer.

        params:
            :dim: Dimensionality of each token
            :mlp_ratio: MLP hidden dimensionality multiplier
        """
        super(CustomBlock, self).__init__()

        self.attention = attention(dim, dropout=drop, num_heads=num_heads)
        self.MLP = mlp(
            dim, dropout=drop, activation_function=activation, mlp_ratio=mlp_ratio
        )
        self.LN_1 = nn.LayerNorm(dim)
        self.LN_2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)
        self.dropPath = DropPath(drop_prob=drop)

    def forward(self, x, mask=None):
        X_a = x + self.dropPath(self.attention(self.LN_1(x), mask=mask))
        X_b = X_a + self.dropPath(self.MLP(self.LN_2(X_a)))
        return X_b


class RobustBlock(nn.Module):
    """
    https://github.com/vtddggg/Robust-Vision-Transformer/blob/main/robust_models.py
    """

    def __init__(
        self,
        dim,
        num_heads,
        drop=0.0,
        mlp_ratio=4.0,
        activation=nn.GELU,
        attention=RobustAttention,
        mlp=RobustMlp,
    ) -> None:
        """
        Transformer encoder block with a custom Attention layer.

        params:
            :dim: Dimensionality of each token
            :mlp_ratio: MLP hidden dimensionality multiplier
        """
        super(RobustBlock, self).__init__()

        self.attention = attention(dim, dropout=drop, num_heads=num_heads)
        self.MLP = mlp(
            dim, dropout=drop, activation_function=activation, mlp_ratio=mlp_ratio
        )
        self.LN_1 = nn.LayerNorm(dim)
        self.LN_2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)
        self.dropPath = DropPath(drop_prob=drop)

    def forward(self, x, mask=None):
        X_a = x + self.dropPath(self.attention(self.LN_1(x), mask=mask))
        X_b = X_a + self.dropPath(self.MLP(self.LN_2(X_a)))
        return X_b


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(PatchEmbed, self).__init__()

        if stride == 2:
            self.avgpool = nn.AvgPool2d(
                (2, 2), stride=2, ceil_mode=True, count_include_pad=False
            )
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, bias=False
            )
            self.norm = nn.BatchNorm2d(out_channels, eps=1e-5)
        elif in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, bias=False
            )
            self.norm = nn.BatchNorm2d(out_channels, eps=1e-5)
        else:
            self.avgpool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.conv(self.avgpool(x)))


class ECBlock(nn.Module):
    """
    Efficient Convolution Block
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        drop=0,
        num_heads=32,
        mlp_ratio=3,
    ):
        super(ECBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert out_channels % num_heads == 0

        self.patch_embed = PatchEmbed(in_channels, out_channels, stride)
        self.MHCA = MultiCHA(out_channels, num_heads, dropout=drop)
        self.attention_path_dropout = DropPath(drop)

        self.conv = LocalityFeedForward(
            out_channels, 1, mlp_ratio, reduction=out_channels
        )

        self.norm = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.attention_path_dropout(self.MHCA(x))
        out = x
        x = x + self.conv(out)  # (B, dim, 14, 14)
        return x


class LTBlock(nn.Module):
    """
    Local Transformer Block
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        drop=0,
        stride=1,
        sr_ratio=1,
        mlp_ratio=2,
        num_heads=32,
        mix_block_ratio=0.75,
    ):
        super(LTBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mix_block_ratio = mix_block_ratio

        self.mhsa_out_channels = _make_divisible(
            int(out_channels * mix_block_ratio), 32
        )
        self.mhca_out_channels = out_channels - self.mhsa_out_channels

        self.patch_embed = PatchEmbed(in_channels, self.mhsa_out_channels, stride)
        self.norm1 = nn.BatchNorm2d(self.mhsa_out_channels, eps=1e-6)

        self.e_mhsa = EMAttention(
            self.mhsa_out_channels,
            num_heads=num_heads,
            drop=drop,
            sr_ratio=sr_ratio,
        )
        self.mhsa_path_dropout = DropPath(drop * mix_block_ratio)

        self.projection = PatchEmbed(
            self.mhsa_out_channels, self.mhca_out_channels, stride=1
        )

        self.mhca = MultiCHA(self.mhca_out_channels, num_heads=num_heads)
        self.mhca_path_dropout = DropPath(drop * (1 - mix_block_ratio))

        self.norm2 = nn.BatchNorm2d(out_channels, eps=1e-6)
        self.conv = LocalityFeedForward(
            out_channels, 1, mlp_ratio, reduction=out_channels
        )

        self.is_bn_merged = False

    def forward(self, x):
        # conv 1x1
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm1(x)
        else:
            out = x
        out = rearrange(out, "b c h w -> b (h w) c")  # b n c

        # ESA
        out = self.mhsa_path_dropout(self.e_mhsa(out))
        x = x + rearrange(out, "b (h w) c -> b c h w", h=H, w=W)

        # conv 1x1
        out = self.projection(x)
        # MCHA
        out = out + self.mhca_path_dropout(self.mhca(out))
        x = torch.cat([x, out], dim=1)

        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm2(x)
        else:
            out = x

        # LFFN
        x = x + self.conv(out)
        return x
