from positional_encodings import build_2d_sincos_posemb
from patch_embeddings import NaivePatchEmbed

import torch
import torch.nn as nn
from einops import rearrange
import ast



class Mlp(nn.Module):
    '''
    Base two layer MLP. From the transformers notebook.
    '''
    def __init__(self, dim, activation_function=nn.GELU, dropout=0.0, mlp_ratio=4.0):
        '''
        params:
            :dim: Dimensionality of each token
            :activation_function: Activation function to use
            :dropout: Dropout rate
            :mlp_ratio: MLP hidden dimensionality multiplier
        '''
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(mlp_ratio * dim)),
            activation_function(),
            nn.Dropout(dropout),
            nn.Linear(int(mlp_ratio * dim), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mlp(x) # returns output of the same dimension as the input


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

        # self.to_qkv = nn.Linear(dim, self.num_heads * self.head_dim * 3, bias=bias)
        self.Q = nn.Linear(dim, dim, bias=False)
        self.K = nn.Linear(dim, dim, bias=False)
        self.V = nn.Linear(dim, dim, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(dropout)

        # Projection
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        '''
        params:
            :x: Input of shape [B N C]. B = batch size, N = sequence length, C = token dimensionality
            :mask: Optional attention mask of shape [B N N]. Wherever it is True, the attention matrix will
            be zero.
            
        returns:
            Output of shape [B N C].
        '''
        B, N, C = x.shape

        # qkv = self.to_qkv(x).chunk(3, dim=-1)
        # q, k, v = map(
        #     lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv
        # )
        q, k, v = self.Q(x), self.K(x), self.V(x)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1,2)

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
            :num_heads: Number of attention heads
            :mlp_ratio: MLP hidden dimensionality multiplier
        """
        super().__init__()

        self.attention = Attention(dim, dropout=drop, num_heads=num_heads)
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


class ViT(nn.Module):
    def __init__(
        self,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        drop_rate=0.0,
        patch_embedding="default",
        positional_encoding=None,
        img_size=(28, 28),
        num_classes=10,
        head_bias=False,
        **kwargs
    ):
        """
        A Vision Transformer for classification.

        params:
            :img_size: Image height and width in pixels
            :patch_size: Patch size height and width in pixels
            :in_channels: Number of input channels
            :embed_dim: Token dimension
            :num_classes: Number of classes
            :depth: Transformer depth
            :num_heads: Number of attention heads
            :mlp_ratio: MLP hidden dimensionality multiplier
        """
        super().__init__()

        # Patch embedding
        if patch_embedding == "default":
            print("Warning: Using default patch embedding.")
            self.patch_embed = NaivePatchEmbed(embed_dim=192, in_channels=1)
        elif patch_embedding["type"] == "NaivePatchEmbedding":
            self.patch_embed = NaivePatchEmbed(**patch_embedding["params"])
        else:
            raise NotImplementedError("Patch embedding not implemented.")
        embed_dim = self.patch_embed.get_embed_dim()

        # Positional encoding
        if positional_encoding == None:
            self.positional_encoding = None
            print("Warning: No positional encoding.")
        elif positional_encoding["type"] == "Fixed2DPositionalEmbedding":
            img_size = ast.literal_eval(positional_encoding["params"]["img_size"])
            self.positional_encoding = nn.Parameter(
                build_2d_sincos_posemb(
                    img_size[0] // self.patch_embed.get_patch_size(), 
                    img_size[1] // self.patch_embed.get_patch_size(), 
                    embed_dim=embed_dim
                ),
                requires_grad=False, #TODO: make this a learnable parameter
            )
        else:
            raise NotImplementedError("Positional encoding not implemented.")

        self.transformer = Transformer(
            dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate
        )

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes, bias=head_bias)
        )

    def forward(self, x):
        proj = self.patch_embed(x)

        if self.positional_encoding is not None:
            proj = proj + self.positional_encoding

        proj = self.transformer(proj)

        pooled = proj.mean(dim=1)
        logits = self.head(pooled)

        return logits
