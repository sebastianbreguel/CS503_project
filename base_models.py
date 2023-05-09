import torch
from torch import nn
from einops import rearrange


# Base self attention layer
class Mlp(nn.Module):
    def __init__(self, dim, act_layer=nn.GELU, drop=0.0, mlp_ratio=4.0):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(mlp_ratio * dim)),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(int(mlp_ratio * dim), dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.mlp(x)


# Base self attention layer
class Attention(nn.Module):
    def __init__(self, dim, bias=False, drop=0.0, num_heads=8):
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

        self.to_qkv = nn.Linear(dim, self.num_heads * self.head_dim * 3, bias=bias)

        self.attend = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(drop)

        # Projection
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(drop))

    def forward(self, x, mask=None):
        B, N, C = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv
        )

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            mask = rearrange(mask, "b n1 n2 -> b 1 n1 n2")
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = self.attend(attn)
        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj(x)

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

        self.attention = Attention(dim, drop=drop, num_heads=num_heads)
        self.MLP = Mlp(dim)
        self.LN_1 = nn.LayerNorm(dim)
        self.LN_2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask=None):
        X_a = x + self.attention(self.LN_1(x), mask=mask)
        X_b = X_a + self.MLP(self.LN_2(X_a))
        return X_b


class Transformer(nn.Module):
    def __init__(
        self, dim, depth, num_heads=8, mlp_ratio=4.0, drop=0.0, masked_block=None
    ):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        self.blocks = nn.ModuleList()

        for _ in range(depth):
            self.blocks.append(
                Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop)
            )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=14, patch_size=2, in_channels=1, embed_dim=192):
        """
        Image to Patch Embedding.

        params:
            :img_size: Image height and width in pixels
            :patch_size: Patch size height and width in pixels
            :in_channels: Number of input channels
            :embed_dim: Token dimension
        """
        super().__init__()

        self.patch_size = patch_size
        self.img_size = img_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.conv = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)

    def forward(self, x):
        return self.conv(x).flatten(2).transpose(1, 2)


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
        x = self.proj(x)
        return x


def build_2d_sincos_posemb(h, w, embed_dim=1024, temperature=10000.0):
    """Sine-cosine positional embeddings from MoCo-v3
    Source: https://github.com/facebookresearch/moco-v3/blob/main/vits.py
    Returns positional embedding of shape [B, N, D]
    """
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
    assert (
        embed_dim % 4 == 0
    ), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature**omega)
    out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
    out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
    pos_emb = torch.cat(
        [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1
    )[None, :, :]
    return pos_emb
