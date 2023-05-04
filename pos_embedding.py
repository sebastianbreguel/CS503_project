import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class Mlp(nn.Module):
    def __init__(self, dim, act_layer=nn.GELU, drop=0.0, mlp_ratio=4.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, int(mlp_ratio * dim))
        self.fc2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.act_layer = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, attn_drop=0.0, proj_drop=0.0, num_heads=8):
        """
        Self-attention layer.

        params:
            :dim: Dimensionality of each token
            :num_heads: Number of attention heads
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.K = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.Q = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.V = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.Wp = nn.Parameter(
            torch.randn(self.num_head * self.head_dim, self.num_head * self.head_dim)
        )

        self.attn_drop = nn.Dropout(attn_drop)

        # Projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        """
        Performs a forward pass through the multi-headed self-attention layer.

        params:
            :x: Input of shape [B N C]. B = batch size, N = sequence length, C = token dimensionality
            :mask: Optional attention mask of shape [B N N]. Wherever it is True, the attention matrix will
            be zero.

        returns:
            Output of shape [B N C].
        """
        B, N, C = x.shape

        q, k, v = (
            self.Q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2),
            self.K(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2),
            self.V(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2),
        )

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = rearrange(mask, "b n1 n2 -> b 1 n1 n2")
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, drop=0.0, mlp_ratio=4.0):
        """
        Transformer encoder block.

        params:
            :dim: Dimensionality of each token
            :num_heads: Number of attention heads
            :mlp_ratio: MLP hidden dimensionality multiplier
        """
        super().__init__()

        # TODO
        self.attention = Attention(dim, num_heads)
        self.MLP = Mlp(dim)
        self.LN_1 = nn.LayerNorm(dim)
        self.LN_2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        """
        Performs a forward pass through the multi-headed self-attention layer.

        params:
            :x: Input of shape [B N C]. B = batch size, N = sequence length, C = token dimensionality
            :mask: Optional attention mask of shape [B N N]. Wherever it is True, the attention matrix will
            be zero.

        returns:
            Output of shape [B N C].
        """
        X_a = x + self.attention(self.LN_1(x), mask=mask)
        X_b = X_a + self.MLP(self.LN_2(X_a))
        return X_b


#
class conv_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding):
        super(conv_embedding, self).__init__()

        self.out_channels = out_channels

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels, 32, kernel_size=(7, 7), stride=(2, 2), padding=(2, 2)
            ),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(32, out_channels, kernel_size=(4, 4), stride=(4, 4)),
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)
        return x
