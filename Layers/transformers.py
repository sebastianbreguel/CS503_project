from collections import OrderedDict


from activations import QuickGELU
from timm.models.layers import DropPath
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


class Mlp(nn.Module):
    """
    Base two layer MLP. From the transformers notebook.
    """

    def __init__(self, dim, activation_function=nn.GELU, dropout=0.0, mlp_ratio=4.0):
        """
        params:
            :dim: Dimensionality of each token
            :activation_function: Activation function to use
            :dropout: Dropout rate
            :mlp_ratio: MLP hidden dimensionality multiplier
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(mlp_ratio * dim)),
            activation_function(),
            nn.Dropout(dropout),
            nn.Linear(int(mlp_ratio * dim), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mlp(x)  # returns output of the same dimension as the input


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False, dropout=0.0):
        """
        Self-attention layer.

        params:
            :dim: Dimensionality of each token
            :num_heads: Number of attention heads
            :bias: Whether to use bias in the linear projection
            :dropout: Dropout rate
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.Q = nn.Linear(dim, dim, bias=False)
        self.K = nn.Linear(dim, dim, bias=False)
        self.V = nn.Linear(dim, dim, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(dropout)

        # Projection
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        """
        params:
            :x: Input of shape [B N C]. B = batch size, N = sequence length, C = token dimensionality
            :mask: Optional attention mask of shape [B N N]. Wherever it is True, the attention matrix will
            be zero.

        returns:
            Output of shape [B N C].
        """
        B, N, C = x.shape

        q, k, v = self.Q(x), self.K(x), self.V(x)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            mask = rearrange(mask, "b n1 n2 -> b 1 n1 n2")
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = self.softmax(attn)
        attn = self.attention_dropout(attn)

        x = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, drop=0.0, mlp_ratio=4.0):
        """
        Transformer encoder block.

        params:
            :dim: Dimensionality of each token
            :mlp_ratio: MLP hidden dimensionality multiplier
        """
        super().__init__()

        self.attention = Attention(dim, dropout=drop, num_heads=num_heads)
        self.MLP = Mlp(dim, dropout=drop)
        self.LN_1 = nn.LayerNorm(dim)
        self.LN_2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask=None):
        X_a = x + self.attention(self.LN_1(x), mask=mask)
        X_b = X_a + self.MLP(self.LN_2(X_a))
        return X_b


class Transformer(nn.Module):
    def __init__(
        self, dim, depth, num_heads=8, mlp_ratio=4.0, drop_rate=0.0, masked_block=None
    ):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        self.blocks = nn.ModuleList()

        for _ in range(depth):
            self.blocks.append(
                Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate)
            )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Parallel_Blocks(nn.Module):
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
        self.Mlps = nn.ModuleList()

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
            self.Mlps.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("norm", nn.LayerNorm(dim)),
                            ("mlp", Mlp(dim, drop_out=drop)),
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
        x = x + sum(attn(x, mask=mask) for attn in self.attns)
        x = x + sum(mlp(x) for mlp in self.mlp)
        return x

    def forward(self, x, mask=None):
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return self._forward_jit(x, mask=mask)
        else:
            return self._forward(x, mask=mask)


class Parallel_transformers(nn.Module):
    def __init__(
        self, dim, depth, num_heads=8, mlp_ratio=4.0, drop_rate=0.0, masked_block=None
    ):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        self.parallel_blocks = 2
        self.depth = depth // self.parallel_blocks
        self.unique = depth % self.parallel_blocks
        self.blocks = nn.ModuleList()

        for _ in range(depth):
            self.blocks.append(
                Parallel_Blocks(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    num_parallel=self.parallel_blocks,
                )
            )

        for _ in range(self.unique):
            self.blocks.append(
                Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate)
            )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ConvAttention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False, dropout=0.0):
        """
        Self-attention layer for Convolutional Proyection

        Oficial implementation of https://arxiv.org/pdf/2103.15808.pdf

        params:
            :dim: Dimensionality of each token
            :num_heads: Number of attention heads
            :bias: Whether to use bias in the linear projection
            :dropout: Dropout rate
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.conv_proj_q = self._build_projection(dim, dim, 3, 2, 2, "dw_bn")
        self.conv_proj_k = self._build_projection(dim, dim, 3, 2, 2, "dw_bn")
        self.conv_proj_v = self._build_projection(dim, dim, 3, 2, 2, "dw_bn")

        self.Q = nn.Linear(dim, dim, bias=False)
        self.K = nn.Linear(dim, dim, bias=False)
        self.V = nn.Linear(dim, dim, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(dropout)

        # Projection
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def _build_projection(self, dim_in, dim_out, kernel_size, padding, stride, method):
        if method == "dw_bn":
            proj = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            nn.Conv2d(
                                dim_in,
                                dim_in,
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride,
                                bias=False,
                                groups=dim_in,
                            ),
                        ),
                        ("bn", nn.BatchNorm2d(dim_in)),
                        ("rearrage", Rearrange("b c h w -> b (h w) c")),
                    ]
                )
            )
        elif method == "avg":
            proj = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "avg",
                            nn.AvgPool2d(
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride,
                                ceil_mode=True,
                            ),
                        ),
                        ("rearrage", Rearrange("b c h w -> b (h w) c")),
                    ]
                )
            )
        elif method == "linear":
            proj = None
        else:
            raise ValueError("Unknown method ({})".format(method))

        return proj

    def forward_conv(self, x):
        x = rearrange(x, "b (h w) c -> b c h w", h=14, w=14)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, "b c h w -> b (h w) c")

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, "b c h w -> b (h w) c")

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, "b c h w -> b (h w) c")

        return q, k, v

    def forward(self, x, mask=None):
        """
        params:
            :x: Input of shape [B N C]. B = batch size, N = sequence length, C = token dimensionality
            :mask: Optional attention mask of shape [B N N]. Wherever it is True, the attention matrix will
            be zero.

        returns:
            Output of shape [B N C].
        """
        B, N, C = x.shape
        if (
            self.conv_proj_q is not None
            or self.conv_proj_k is not None
            or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x)

        q, k, v = self.Q(x), self.K(x), self.V(x)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            mask = rearrange(mask, "b n1 n2 -> b 1 n1 n2")
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = self.softmax(attn)
        attn = self.attention_dropout(attn)

        x = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj(x)

        return x


class ConvBlock(nn.Module):
    def __init__(self, dim, num_heads, drop=0.0, mlp_ratio=4.0):
        """
        Transformer encoder block.

        params:
            :dim: Dimensionality of each token
            :mlp_ratio: MLP hidden dimensionality multiplier
        """
        super().__init__()

        self.attention = ConvAttention(dim, dropout=drop, num_heads=num_heads)
        self.MLP = Mlp(dim)
        self.LN_1 = nn.LayerNorm(dim)
        self.LN_2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask=None):
        X_a = x + self.attention(self.LN_1(x), mask=mask)
        X_b = X_a + self.MLP(self.LN_2(X_a))
        return X_b


class Custom_transformer(nn.Module):

    """
    Class to use custom blocks in the transformer modifications
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.0,
        masked_block=None,
        block=ConvBlock,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        self.blocks = nn.ModuleList()

        for _ in range(depth):
            self.blocks.append(
                block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate)
            )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
