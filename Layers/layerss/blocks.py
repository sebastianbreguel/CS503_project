from collections import OrderedDict

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


# import drop path from folder layers
from Layers.utils import DropPath
from .attention import Attention, RobustAttention
from .mlp import Mlp


class Block(nn.Module):
    def __init__(self, dim, num_heads, drop=0.0, mlp_ratio=4.0, activation=nn.GELU):
        """
        Transformer encoder block.

        params:
            :dim: Dimensionality of each token
            :mlp_ratio: MLP hidden dimensionality multiplier
        """
        super().__init__()

        self.attention = Attention(dim, dropout=drop, num_heads=num_heads)
        self.MLP = Mlp(
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

    def __init__(self, dim, num_heads, drop=0.0, mlp_ratio=4.0, num_parallel=2):
        """
        Transformer encoder block.

        params:
            :dim: Dimensionality of each token
            :num_heads: Number of attention heads
            :mlp_ratio: MLP hidden dimensionality multiplier
            :num_parallel: Number of parallel blocks
        """
        super().__init__()
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
                            ("ls", nn.Identity()),
                            ("drop_path", nn.Identity()),
                        ]
                    )
                )
            )
            self.mlps.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("norm", nn.LayerNorm(dim)),
                            ("mlp", Mlp(dim, dropout=drop)),
                            ("ls", nn.Identity()),
                            ("drop_path", nn.Identity()),
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


class ConvBlock(nn.Module):
    def __init__(self, dim, num_heads, drop=0.0, mlp_ratio=4.0):
        """
        Transformer encoder block.

        params:
            :dim: Dimensionality of each token
            :mlp_ratio: MLP hidden dimensionality multiplier
        """
        super().__init__()

        self.attention = RobustAttention(dim, dropout=drop, num_heads=num_heads)
        self.MLP = Mlp(dim)
        self.LN_1 = nn.LayerNorm(dim)
        self.LN_2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask=None):
        X_a = x + self.attention(self.LN_1(x), mask=mask)
        X_b = X_a + self.MLP(self.LN_2(X_a))
        return X_b
