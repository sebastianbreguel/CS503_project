from collections import OrderedDict


from .activations import QuickGELU

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .activations import DropPath


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


class Parallel_transformers(nn.Module):
    def __init__(
        self, dim, depth, num_heads=8, mlp_ratio=4.0, drop_rate=0.0, masked_block=None
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.parallel_blocks = 3
        self.depth = depth // self.parallel_blocks
        self.unique = depth % self.parallel_blocks
        self.blocks = nn.ModuleList()

        for _ in range(depth):
            self.blocks.append(
                Parallel_blocks(
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
        Self-attention layer for Convolutional Proyection https://arxiv.org/pdf/2103.15808.pdf
        Source: https://github.com/leoxiaobin/CvT/tree/main

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

        self.attention = RobustAttention(dim, dropout=drop, num_heads=num_heads)
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


class SpatialDepthWisePerHeadConvolution(nn.Module):
    """
    ## Spatial Depth Wise Per Head Convolution
    """

    def __init__(self, heads: int, d_k: int, kernel_size: int = 3):
        """
        * `heads` is the number of heads
        * `d_k` is the number of channels in each head
        """
        super().__init__()
        self.kernel_size = kernel_size
        # We use PyTorch's `Conv1d` module.
        # We set the number of groups to be equal to the number of channels from each head
        # so that it does a separate convolution
        # (with different kernels) for each channel and head.
        # We add padding to both sides and later crop the right most `kernel_size - 1` results
        self.conv = nn.Conv1d(
            in_channels=d_k * heads,
            out_channels=d_k * heads,
            kernel_size=(kernel_size,),
            padding=(kernel_size - 1,),
            groups=d_k * heads,
        )

    def forward(self, x: torch.Tensor):
        """
        `x` has shape `[seq_len, batch_size, heads, d_k]`
        """

        # Get the shape
        seq_len, batch_size, heads, d_k = x.shape
        # Permute to `[batch_size, heads, d_k, seq_len]`
        x = x.permute(1, 2, 3, 0)
        # Change the shape to `[batch_size heads * d_k, seq_len]`
        x = x.view(batch_size, heads * d_k, seq_len)

        # 1D convolution accepts input of the form `[N, channels, sequence]`
        x = self.conv(x)
        # Crop the right most `kernel_size - 1` results since we padded both sides
        x = x[:, :, : -(self.kernel_size - 1)]
        # Reshape to `[batch_size, heads, d_k, seq_len]`
        x = x.view(batch_size, heads, d_k, seq_len)
        # Permute to `[seq_len, batch_size, heads, d_k]`
        x = x.permute(3, 0, 1, 2)

        #
        return x


class PrepareForMultiHeadAttention(nn.Module):
    """
    <a id="PrepareMHA"></a>

    ## Prepare for multi-head attention

    This module does a linear transformation and splits the vector into given
    number of heads for multi-head attention.
    This is used to transform **key**, **query**, and **value** vectors.
    """

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        # Linear layer for linear transform
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        # Number of heads
        self.heads = heads
        # Number of dimensions in vectors in each head
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        # Input has shape `[seq_len, batch_size, d_model]` or `[batch_size, d_model]`.
        # We apply the linear transformation to the last dimension and split that into
        # the heads.
        head_shape = x.shape[:-1]

        # Linear transform
        x = self.linear(x)

        # Split last dimension into heads
        x = x.view(*head_shape, self.heads, self.d_k)

        # Output has shape `[seq_len, batch_size, heads, d_k]` or `[batch_size, heads, d_model]`
        return x


class MultiDPHConvHeadAttention(nn.Module):
    """
    ## Multi-per-Head-Depth-wise-Conv-Head Attention

    Implementation of the paper: https://arxiv.org/pdf/2109.08668.pdf
    source: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/primer_ez/variations.py
    """

    def __init__(self, dim, num_heads=8, bias=False, dropout=0.0):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # self.scale = 1 / math.sqrt(self.d_k)
        self.scale = self.head_dim**-0.5

        self.Q = nn.Sequential(
            PrepareForMultiHeadAttention(
                dim, self.num_heads, self.head_dim, bias=False
            ),
            SpatialDepthWisePerHeadConvolution(self.num_heads, self.head_dim),
        )
        self.K = nn.Sequential(
            PrepareForMultiHeadAttention(
                dim, self.num_heads, self.head_dim, bias=False
            ),
            SpatialDepthWisePerHeadConvolution(self.num_heads, self.head_dim),
        )
        self.V = nn.Sequential(
            PrepareForMultiHeadAttention(dim, self.num_heads, self.head_dim, bias=True),
            SpatialDepthWisePerHeadConvolution(self.num_heads, self.head_dim),
        )

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

        q = self.Q(x).transpose(1, 2)
        k = self.K(x).transpose(1, 2)
        v = self.V(x).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # scores = self.get_scores(q, k) * self.scale

        attn = self.softmax(scores)
        attn = self.attention_dropout(attn)
        x = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj(x)

        return x


class RobustAttention(nn.Module):

    """
    Implementation of the Position-Aware Attention Scaling https://arxiv.org/pdf/2105.07926.pdf
    source: https://github.com/vtddggg/Robust-Vision-Transformer/tree/main
    """

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
        self.W = nn.Parameter(
            torch.randn(256, 256), requires_grad=True  # TODO see how to obtain the N
        )  # learnable paramter to learn the position encoding

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

        attn = torch.matmul(q, k.transpose(-1, -2))

        # To make the original rescaling process of dotproduct attention position-aware, we define a learnable position importance matrix Wp ∈ R
        # NxN , which presents the importance of each pair of q-k.
        attn = attn * (self.W * self.scale)

        if mask is not None:
            mask = rearrange(mask, "b n1 n2 -> b 1 n1 n2")
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = self.softmax(attn)
        attn = self.attention_dropout(attn)

        x = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj(x)

        return x
