import ast
import math

import torch
import torch.nn as nn
from einops import rearrange

from Layers import (
    ECBlock,
    LTBlock,
    BasicStem,
    ConvEmbedding,
    CustomTransformer,
    GraphPatchEmbed,
    LayerNorm,
    MedVitTransformer,
    NaivePatchEmbed,
    RVTransformer,
    SineCosinePosEmbedding,
    Transformer,
)


class ViT(nn.Module):
    """
    Original Vision Transformer Model   (https://arxiv.org/pdf/2010.11929.pdf).
    """

    def __init__(self, depth=4, num_heads=4, mlp_ratio=4.0, drop_rate=0.0, patch_embedding="default", positional_encoding=None, img_size=(28, 28), num_classes=10, head_bias=False, **kwargs):
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
        super(ViT, self).__init__()

        # Patch embedding
        if patch_embedding == "default":
            print("Warning: Using default patch embedding.")
            self.patch_embed = NaivePatchEmbed(embed_dim=192, in_channels=1)
        elif patch_embedding["type"] == "NaivePatchEmbedding":
            self.patch_embed = NaivePatchEmbed(**patch_embedding["params"])
        elif patch_embedding["type"] == "GraphPatchEmbedding":
            self.patch_embed = GraphPatchEmbed(**patch_embedding["params"])
        else:
            raise NotImplementedError("Patch embedding not implemented.")
        embed_dim = self.patch_embed.get_embed_dim()

        # Positional encoding
        if positional_encoding == None:
            self.positional_encoding = None
            print("Warning: No positional encoding.")

        elif positional_encoding["type"] == "Fixed2DPositionalEmbedding":
            img_size = ast.literal_eval(positional_encoding["params"]["img_size"])
            self.positional_encoding = SineCosinePosEmbedding(
                img_size[0] // self.patch_embed.get_patch_size(),
                img_size[1] // self.patch_embed.get_patch_size(),
                embed_dim=embed_dim,
                requires_grad=False,
            )

        else:
            raise NotImplementedError("Positional encoding not implemented.")

        self.transformer = Transformer(
            dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
        )

        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes, bias=head_bias))

    def forward(self, x):
        proj = self.patch_embed(x)

        if self.positional_encoding is not None:
            proj = proj + self.positional_encoding(proj)

        proj = self.transformer(proj)

        pooled = proj.mean(dim=1)
        logits = self.head(pooled)

        return logits


class BreguiT(nn.Module):
    """
    Personal version of Vision transformers
    """

    def __init__(
        self, depth=4, num_heads=4, mlp_ratio=4.0, drop_rate=0.0, patch_embedding="default", positional_encoding=None, img_size=(28, 28), num_classes=10, head_bias=False, preLayerNorm=False, **kwargs
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
        super(BreguiT, self).__init__()

        img_size = ast.literal_eval(positional_encoding["params"]["img_size"])
        self.PrelayerNorm = LayerNorm([patch_embedding["params"]["in_channels"], img_size[0], img_size[1]]) if preLayerNorm else nn.Identity()

        # Patch embedding
        if patch_embedding == "default":
            print("Warning: Using default patch embedding.")
            self.patch_embed = NaivePatchEmbed(embed_dim=192, in_channels=1)
        elif patch_embedding["type"] == "NaivePatchEmbedding":
            self.patch_embed = NaivePatchEmbed(**patch_embedding["params"])

        elif patch_embedding["type"] == "ConvEmbedding":
            self.patch_embed = ConvEmbedding(**patch_embedding["params"])
        else:
            raise NotImplementedError("Patch embedding not implemented.")
        embed_dim = self.patch_embed.get_embed_dim()

        # Positional encoding
        if positional_encoding == None:
            self.positional_encoding = None
            print("Warning: No positional encoding.")
        elif positional_encoding["type"] == "Fixed2DPositionalEmbedding":
            img_size = ast.literal_eval(positional_encoding["params"]["img_size"])
            self.positional_encoding = SineCosinePosEmbedding(
                img_size[0] // self.patch_embed.get_patch_size(),
                img_size[1] // self.patch_embed.get_patch_size(),
                embed_dim=embed_dim,
                requires_grad=positional_encoding["params"]["requires_grad"],
            )
        else:
            raise NotImplementedError("Positional encoding not implemented.")
        self.transformer = CustomTransformer(
            dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
        )
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes, bias=head_bias))

    def forward(self, x):
        # x = self.PrelayerNorm(x)

        proj = self.patch_embed(x)

        if self.positional_encoding is not None:
            proj = proj + self.positional_encoding(proj)

        proj = self.transformer(proj)

        pooled = proj.mean(dim=1)
        logits = self.head(pooled)

        return logits


class RVT(nn.Module):
    """
    Robust Vision Transformerhttps://arxiv.org/pdf/2105.07926.pdf
    source: https://github.com/vtddggg/Robust-Vision-Transformer/tree/main
    """

    def __init__(
        self, depth=4, num_heads=4, mlp_ratio=4.0, drop_rate=0.0, patch_embedding="default", positional_encoding=None, img_size=(28, 28), num_classes=10, head_bias=False, preLayerNorm=False, **kwargs
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
        super(RVT, self).__init__()

        img_size = ast.literal_eval(positional_encoding["params"]["img_size"])
        self.PrelayerNorm = LayerNorm([patch_embedding["params"]["in_channels"], img_size[0], img_size[1]]) if preLayerNorm else nn.Identity()

        # Patch embedding
        if patch_embedding == "default":
            print("Warning: Using default patch embedding.")
            self.patch_embed = NaivePatchEmbed(embed_dim=192, in_channels=1)
        elif patch_embedding["type"] == "NaivePatchEmbedding":
            self.patch_embed = NaivePatchEmbed(**patch_embedding["params"])

        elif patch_embedding["type"] == "ConvEmbedding":
            self.patch_embed = ConvEmbedding(**patch_embedding["params"])
        else:
            raise NotImplementedError("Patch embedding not implemented.")
        embed_dim = self.patch_embed.get_embed_dim()

        # Positional encoding
        if positional_encoding == None:
            self.positional_encoding = None
            print("Warning: No positional encoding.")
        elif positional_encoding["type"] == "Fixed2DPositionalEmbedding":
            img_size = ast.literal_eval(positional_encoding["params"]["img_size"])
            self.positional_encoding = SineCosinePosEmbedding(
                img_size[0] // self.patch_embed.get_patch_size(),
                img_size[1] // self.patch_embed.get_patch_size(),
                embed_dim=embed_dim,
                requires_grad=positional_encoding["params"]["requires_grad"],
            )
        else:
            raise NotImplementedError("Positional encoding not implemented.")
        self.transformer = CustomTransformer(
            dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
        )
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes, bias=head_bias))

        self.pooling = nn.AvgPool2d(1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x = self.PrelayerNorm(x)

        proj = self.patch_embed(x)

        if self.positional_encoding is not None:
            proj = proj + self.positional_encoding(proj)

        proj = self.transformer(proj)
        proj = self.norm(self.pooling(proj).squeeze())

        pooled = proj.mean(dim=1)
        logits = self.head(pooled)

        return logits


class MedViT(nn.Module):
    """
    MedViT: A Robust Vision Transformer for Generalized Medical Image Classification https://arxiv.org/abs/2302.09462
    -source: https://github.com/Omid-Nejati/MedViT/blob/main/MedViT.py
    """

    def __init__(
        self,
        stem_chs,
        depths,
        drop_rate,
        num_classes=100,
        strides=[1, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        num_heads=32,
        mix_block_ratio=0.75,
    ):
        super(MedViT, self).__init__()

        self.stem = BasicStem(3, out_ch=stem_chs[0])

        input_channel = stem_chs[-1]

        ecbs = 1
        ltbs = 1

        self.features = []
        for stage_id in range(len(depths)):
            for ecb_id in range(ecbs):
                self.features.append(
                    ECBlock(
                        in_channels=input_channel,
                        out_channels=input_channel,
                        drop=drop_rate,
                        mlp_ratio=4,
                        stride=1,
                    )
                )
            for ltb_id in range(ltbs):
                self.features.append(
                    LTBlock(
                        in_channels=input_channel,
                        out_channels=input_channel,
                        sr_ratio=2,
                        drop=drop_rate,
                        mlp_ratio=4,
                        num_heads=num_heads,
                        stride=1,
                    )
                )

        self.features = nn.Sequential(*self.features)

        self.norm = nn.BatchNorm2d(input_channel, eps=1e-6)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj_head = nn.Sequential(
            nn.Linear(input_channel, num_classes),
        )

        self.stage_out_idx = [sum(depths[: idx + 1]) - 1 for idx in range(len(depths))]
        print("initialize_weights...")

    def forward(self, x):
        x = self.stem(x)
        for layer in self.features:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.proj_head(x)
        return x
