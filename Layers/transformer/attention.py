import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from Layers.helper import SELayer, _build_projection, h_swish
from Layers.positional_encodings import RelativePos


class Attention(nn.Module):
    """
    Original Self-Attention Layer from https://arxiv.org/pdf/1706.03762.pdf

    params:
        :dim: Dimensionality of each token
        :num_heads: Number of attention heads
        :bias: Whether to use bias in the linear projection
        :dropout: Dropout rate
    """

    def __init__(self, dim, num_heads=8, bias=False, dropout=0.0) -> None:
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.Q = nn.Linear(dim, dim, bias)
        self.K = nn.Linear(dim, dim, bias)
        self.V = nn.Linear(dim, dim, bias)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # Projection
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.tensor, mask=None) -> torch.tensor:
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

        attn = self.dropout(self.softmax(attn))

        x = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj(x)

        return x


# Convolutional Proyection


# convolution pre proyection
class ConvAttention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False, dropout=0.0, size=14) -> None:
        """
        Self-attention layer for Convolutional Proyection https://arxiv.org/pdf/2103.15808.pdf
        - Source: https://github.com/leoxiaobin/CvT/tree/main

        params:
            :dim: Dimensionality of each token
            :num_heads: Number of attention heads
            :bias: Whether to use bias in the linear projection
            :dropout: Dropout rate
        """
        super(ConvAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.conv_proj_q = _build_projection(dim, dim, 3, 1, 1, "dw_bn")
        self.conv_proj_k = _build_projection(dim, dim, 3, 1, 1, "dw_bn")
        self.conv_proj_v = _build_projection(dim, dim, 3, 1, 1, "dw_bn")

        self.Q = nn.Linear(dim, dim, bias)
        self.K = nn.Linear(dim, dim, bias)
        self.V = nn.Linear(dim, dim, bias)
        self.size = size

        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(dropout)

        # Projection
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward_conv(self, x):
        x = rearrange(x, "b (h w) c -> b c h w", h=self.size, w=self.size)

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

    def forward(self, x: torch.tensor, mask=None) -> torch.tensor:
        """
        params:
            :x: Input of shape [B N C]. B = batch size, N = sequence length, C = token dimensionality
            :mask: Optional attention mask of shape [B N N]. Wherever it is True, the attention matrix will
            be zero.

        returns:
            Output of shape [B N C].
        """
        B, N, C = x.shape
        if self.conv_proj_q is not None or self.conv_proj_k is not None or self.conv_proj_v is not None:
            q, k, v = self.forward_conv(x)
            q, k, v = self.Q(q), self.K(k), self.V(v)
        else:
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


# proyecyion and then convolution
class SpatialDepthWisePerHeadConvolution(nn.Module):
    """
    Spatial Depth Wise Per Head Convolution https://arxiv.org/pdf/2109.08668.pdf
    - source: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/primer_ez/variations.py
    """

    def __init__(self, heads: int, dim: int, kernel_size: int = 3) -> None:
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

        x = self.conv(x)
        # Crop the right most `kernel_size - 1` results since we padded both sides
        x = x[:, :, : -(self.kernel_size - 1)]
        x = x.view(B, self.heads, self.head_dim, N).permute(3, 0, 1, 2).to(memory_format=torch.contiguous_format)

        return x


class MultiDPHConvHeadAttention(nn.Module):
    """
    Multi-per-Head-Depth-wise-Conv-Head Attention  https://arxiv.org/pdf/2109.08668.pdf
    - source: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/primer_ez/variations.py
    """

    def __init__(self, dim, num_heads=8, bias=False, dropout=0.0) -> None:
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

    def forward(self, x: torch.tensor, mask=None) -> torch.tensor:
        """
        params:
            :x: Input of shape [B N C]. B = batch size, N = sequence length, C = token dimensionality
            :mask: Optional attention mask of shape [B N N]. Wherever it is True, the attention matrix will
            be zero.

        returns:
            Output of shape [B N C].
        """
        B, N, C = x.shape
        q = self.Q(self.Q_linear(x).view(B, self.num_heads * self.head_dim, N)).transpose(1, 2)
        k = self.K(self.K_linear(x).view(B, self.num_heads * self.head_dim, N)).transpose(1, 2)
        v = self.V(self.V_linear(x).view(B, self.num_heads * self.head_dim, N)).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.softmax(scores)
        attn = self.attention_dropout(attn)
        x = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj(x)

        return x


# Av pool on qk
class EMAttention(nn.Module):
    """
    Efficient Multi-head Attention https://arxiv.org/pdf/2105.13677.pdf
    -source: https://github.com/wofmanaf/ResT/blob/main/models/rest.py
    """

    def __init__(
        self,
        dim,
        num_heads=32,
        bias=False,
        drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.Q = nn.Linear(dim, dim, bias=bias)
        self.K = nn.Linear(dim, dim, bias=bias)
        self.V = nn.Linear(dim, dim, bias=bias)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(drop))
        self.attn_drop = nn.Dropout(drop)
        self.softmax = nn.Softmax(dim=-1)

        self.sr_ratio = sr_ratio
        #!
        self.N_ratio = sr_ratio**2
        if sr_ratio > 1:
            self.sr = nn.AvgPool1d(kernel_size=self.N_ratio, stride=self.N_ratio)
            self.sr_norm = nn.BatchNorm1d(dim, eps=1e-6)

    def forward(self, x):
        B, N, C = x.shape

        q = self.Q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if self.sr_ratio > 1:
            x_ = self.sr(x.transpose(1, 2))
            x_ = self.sr_norm(x_)
            x = x_.transpose(1, 2)

        k, v = self.K(x), self.V(x)

        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj(x)

        return x


# Multi-Head Convolution Attention
class MultiCHA(nn.Module):
    """
    Multi-Head Convolutional Attention
    """

    def __init__(self, out_channels, num_heads, dropout=0.0):
        super(MultiCHA, self).__init__()

        self.group_conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=out_channels // num_heads)
        self.norm = nn.BatchNorm2d(out_channels, eps=1e-6)
        self.act = nn.ReLU(inplace=True)
        self.projection = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.projection(out)
        out = self.dropout(out)
        return out


class LocalityFeedForward(nn.Module):
    """
    Localy FNN https://arxiv.org/pdf/2301.11553.pdf
    - source: https://github.com/Omid-Nejati/MedViT/blob/main/MedViT.py
    """

    def __init__(
        self,
        dim,
        stride,
        expand_ratio=4.0,
        reduction=4,
    ):
        """
        :param dim: the input/output dimension
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.
        :param act: the activation function.
                    relu: ReLU
                    hs+se: h_swish and SE module
        :param reduction: reduction rate in SE module.
        """
        super(LocalityFeedForward, self).__init__()
        hidden_dim = int(dim * expand_ratio)
        kernel_size = 3

        self.conv = nn.Sequential(
            # the first 1x1 convolution.
            nn.Conv2d(dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            h_swish(),
            # the depth-wise convolution between the two linear layers
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size,
                stride,
                kernel_size // 2,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            h_swish(),
            SELayer(hidden_dim, reduction=reduction),
            # the second
            #  1x1 convolution.
            nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        x = x + self.conv(x)
        return x


######################################
### RELATIVE POSITION ATTENTIONS  ###
######################################


class AxialAttention(nn.Module):
    """
    Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation https://arxiv.org/pdf/2003.07853.pdf
    """

    def __init__(self, dim, num_heads=8, bias=False, dropout=0.0) -> None:
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

        self.Q = nn.Linear(dim, dim, bias)
        self.K = nn.Linear(dim, dim, bias)
        self.V = nn.Linear(dim, dim, bias)

        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(dropout)
        self.max_length = 1028
        self.Eq = RelativePos(dim, self.num_heads, self.head_dim)
        self.Ek = RelativePos(dim, self.num_heads, self.head_dim)
        self.Ev = RelativePos(dim, self.num_heads, self.head_dim)

        # Projection
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.tensor, mask=None) -> torch.tensor:
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
        attn = torch.matmul(q, k.transpose(-1, -2))

        attn = (attn + QEr + KEr) * self.scale

        if mask is not None:
            mask = rearrange(mask, "b n1 n2 -> b 1 n1 n2")
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = self.softmax(attn)
        attn = self.attention_dropout(attn)

        VEr = self.Ev(attn, N, transpose=False)
        sv = torch.matmul(attn, v)

        x = sv + VEr
        x = x.transpose(1, 2).contiguous().view(B, N, C)

        x = self.proj(x)

        return x


class RoformerAttention(nn.Module):
    """
     RoFormer: Enhanced Transformer with Rotary Position Embedding https://arxiv.org/pdf/2104.09864.pdf
    - Source: https://github.com/singaln/Roformer_Simlarity/blob/master/Rotransformer.py
    """

    def __init__(self, dim, num_heads=8, bias=False, dropout=0.0) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.Q = nn.Linear(dim, dim, bias)
        self.K = nn.Linear(dim, dim, bias)
        self.V = nn.Linear(dim, dim, bias)

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def sinusoidal_position_embeddings(self, inputs):
        output_dim = self.dim // self.num_heads
        seq_len = inputs.size(1)
        position_ids = torch.arange(0, seq_len, dtype=torch.float32, device=inputs.device)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float32)
        indices = torch.pow(10000.0, -2 * indices / output_dim).to(inputs.device)
        embeddings = torch.einsum("n,d->nd", position_ids, indices)
        embeddings = torch.stack([embeddings.sin(), embeddings.cos()], dim=-1)
        embeddings = torch.reshape(embeddings, (seq_len, output_dim))
        embeddings = embeddings[None, None, :, :]
        cos_pos = embeddings[..., 1::2]
        sin_pos = embeddings[..., ::2]
        cos_pos = cos_pos.view

        cos_pos = torch.repeat_interleave(embeddings[..., 1::2], 2, dim=-1)
        sin_pos = torch.repeat_interleave(embeddings[..., ::2], 2, dim=-1)
        return cos_pos, sin_pos

    def embed_value(self, x, cos_pos, sin_pos):
        xw2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)
        x = x * cos_pos + xw2 * sin_pos
        return x

    def forward(self, x, mask=None):
        B, N, C = x.shape

        q, k, v = self.Q(x), self.K(x), self.V(x)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        cos_pos, sin_pos = self.sinusoidal_position_embeddings(x)
        q = self.embed_value(q, cos_pos, sin_pos)
        k = self.embed_value(k, cos_pos, sin_pos)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            attn = attn + mask

        attn = self.dropout(self.softmax(attn))

        attn = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, C)

        return attn


class RobustAttention(nn.Module):

    """
    Position-Aware Attention Scaling https://arxiv.org/pdf/2105.07926.pdf
    - source: https://github.com/vtddggg/Robust-Vision-Transformer/tree/main
    """

    def __init__(self, dim, num_heads=8, bias=False, dropout=0.0, size=14) -> None:
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

        self.Q = nn.Linear(dim, dim, bias)
        self.K = nn.Linear(dim, dim, bias)
        self.V = nn.Linear(dim, dim, bias)
        self.size = size * size
        self.W = nn.Parameter(torch.randn(self.size, self.size), requires_grad=True)  # TODO see how to obtain the N  # learnable paramter to learn the position encoding

        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(dropout)

        # Projection
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.tensor, mask=None) -> torch.tensor:
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


######################################
### RELATIVE POSITION ATTENTIONS  ###
######################################
