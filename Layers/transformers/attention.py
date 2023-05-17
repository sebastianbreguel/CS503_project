from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from Layers.positional_encodings import Relative2DPosEncQKV, relativePos


# Baseline
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False, dropout=0.0):
        """
        Original Self-Attention Layer from https://arxiv.org/pdf/1706.03762.pdf

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


# Convolutional Proyection
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


class SpatialDepthWisePerHeadConvolution(nn.Module):
    """
    ## Spatial Depth Wise Per Head Convolution


    Implementation of the paper: https://arxiv.org/pdf/2109.08668.pdf
    source: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/primer_ez/variations.py
    """

    def __init__(self, heads: int, dim: int, kernel_size: int = 3):
        """
        * `heads` is the number of heads
        * `d_k` is the number of channels in each head
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.heads = heads
        self.head_dim = dim
        self.conv = nn.Conv1d(
            in_channels=dim * heads,
            out_channels=dim * heads,
            kernel_size=(kernel_size,),
            padding=(kernel_size - 1,),
            groups=dim * heads,
        )

    def forward(self, x: torch.Tensor):
        """
        params:
            :x: Input of shape [B (H * D) N]., N = sequence length, B = batch size H = token head, D = token dimensionality

        returns:
            Output of shape [N B H D].-

        """

        B, _, N = x.shape

        # 1D convolution accepts input of the form `[N, channels, sequence]`
        x = self.conv(x)

        # Crop the right most `kernel_size - 1` results since we padded both sides
        x = x[:, :, : -(self.kernel_size - 1)]
        x = x.view(B, self.heads, self.head_dim, N).permute(3, 0, 1, 2)

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

        self.Q_linear = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.K_linear = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.V_linear = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)

        self.Q = SpatialDepthWisePerHeadConvolution(self.num_heads, self.head_dim)
        self.K = SpatialDepthWisePerHeadConvolution(self.num_heads, self.head_dim)
        self.V = SpatialDepthWisePerHeadConvolution(self.num_heads, self.head_dim)

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
        q = self.Q(
            self.Q_linear(x).view(B, self.num_heads * self.head_dim, N)
        ).transpose(1, 2)
        k = self.K(
            self.K_linear(x).view(B, self.num_heads * self.head_dim, N)
        ).transpose(1, 2)
        v = self.V(
            self.V_linear(x).view(B, self.num_heads * self.head_dim, N)
        ).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

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


# Baseline
class AxialAttention(nn.Module):
    """
    Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation

    """

    def __init__(self, dim, num_heads=8, bias=False, dropout=0.0):
        """

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
        self.max_length = 1028
        self.Eq = relativePos(dim, self.num_heads, self.head_dim)
        self.Ek = relativePos(dim, self.num_heads, self.head_dim)
        self.Ev = relativePos(dim, self.num_heads, self.head_dim)

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

        QEr = self.Eq(q, N)
        KEr = self.Ek(k, N)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # do the element wise sum
        attn = attn + QEr + KEr

        if mask is not None:
            mask = rearrange(mask, "b n1 n2 -> b 1 n1 n2")
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = self.softmax(attn)
        attn = self.attention_dropout(attn)

        VEr = self.Ev(attn, N, transpose=False)
        sv = torch.matmul(attn, v)

        # do the element wise sum
        x = sv + VEr

        x = x.transpose(1, 2).contiguous().view(B, N, C)

        x = self.proj(x)

        return x

    def _skew(self, qe):
        # pad a column of zeros on the left
        padded_qe = F.pad(qe, [1, 0])
        s = padded_qe.shape
        padded_qe = padded_qe.view(s[0], s[1], s[3], s[2])
        # take out first (padded) row
        return padded_qe[:, :, 1:, :]
