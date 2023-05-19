import ast
import math

import torch
import torch.nn as nn
from einops import rearrange

from Layers import (
    ConvEmbedding,
    CustomTransformer,
    LayerNorm,
    NaivePatchEmbed,
    SineCosinePosEmbedding,
    Transformer,
    RVTransformer,
)


class ViT(nn.Module):
    """
    Original Vision Transformer Model   (https://arxiv.org/pdf/2010.11929.pdf).
    """

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

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes, bias=head_bias)
        )

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
        preLayerNorm=False,
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

        img_size = ast.literal_eval(positional_encoding["params"]["img_size"])
        self.PrelayerNorm = (
            LayerNorm(
                [patch_embedding["params"]["in_channels"], img_size[0], img_size[1]]
            )
            if preLayerNorm
            else nn.Identity()
        )

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
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes, bias=head_bias)
        )

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
    Personal version of Vision transformers
    """

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
        preLayerNorm=False,
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

        img_size = ast.literal_eval(positional_encoding["params"]["img_size"])
        self.PrelayerNorm = (
            LayerNorm(
                [patch_embedding["params"]["in_channels"], img_size[0], img_size[1]]
            )
            if preLayerNorm
            else nn.Identity()
        )

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
        print(embed_dim)

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
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes, bias=head_bias)
        )

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


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels, eps=1e-6)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(PatchEmbed, self).__init__()
        if stride == 2:
            self.avgpool = nn.AvgPool2d(
                (2, 2), stride=2, ceil_mode=True, count_include_pad=False
            )
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, bias=False
            )
            self.norm = nn.BatchNorm2d(out_channels, eps=1e-6)
        elif in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, bias=False
            )
            self.norm = nn.BatchNorm2d(out_channels, eps=1e-6)
        else:
            self.avgpool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.conv(self.avgpool(x)))


class MHCA(nn.Module):
    """
    Multi-Head Convolutional Attention
    """

    def __init__(self, out_channels, head_dim):
        super(MHCA, self).__init__()
        self.group_conv3x3 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_channels // head_dim,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels, eps=1e-6)
        self.act = nn.ReLU(inplace=True)
        self.projection = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.projection(out)
        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class MedViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNReLU(3, 64, kernel_size=3, stride=2),
            ConvBNReLU(64, 32, kernel_size=3, stride=1),
            ConvBNReLU(32, 64, kernel_size=3, stride=1),
            ConvBNReLU(64, 64, kernel_size=3, stride=2),
        )
