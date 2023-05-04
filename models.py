from base_models import build_2d_sincos_posemb, Transformer
from Modifications import PatchEmbed
import torch
import torch.nn as nn


class ViT(nn.Module):
    def __init__(
        self,
        img_size=14,
        patch_size=2,
        in_channels=1,
        embed_dim=192,
        num_classes=10,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        head_bias=False,
        drop=0.0,
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

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(
            build_2d_sincos_posemb(
                img_size // patch_size, img_size // patch_size, embed_dim=embed_dim
            ),
            requires_grad=False,
        )

        self.transformer = Transformer(
            dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio
        )

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes, bias=head_bias)
        )

    def forward(self, x):
        proj = self.patch_embed(x)
        proj = proj + self.pos_embed

        proj = self.transformer(proj)

        pooled = proj.mean(dim=1)
        logits = self.head(pooled)

        return logits
