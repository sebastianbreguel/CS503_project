import math

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

import torch_geometric
from torch_geometric.nn import GCNConv


class NaivePatchEmbed(nn.Module):
    """
    Basic Patch Embedding Module. Same as in the transformers graded notebook.
    """

    def __init__(
        self, patch_size=2, in_channels=1, embed_dim=192, norm_layer=None
    ) -> None:
        """
        Image to Patch Embedding.

        params:
            :patch_size: Patch size height and width in pixels
            :in_channels: Number of input channels
            :embed_dim: Token dimension
        """
        super(NaivePatchEmbed, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.norm_layer = norm_layer(embed_dim) if norm_layer else None

    def get_patch_size(self):
        return self.patch_size

    def get_embed_dim(self):
        return self.embed_dim

    def forward(self, x):
        """
        params:
            :x: Input of shape [B C H W]. B = batch size, C = number of channels, H = image height, W = image width
        returns:
            Output of shape [B N C].
        """

        x = self.conv(x).flatten(2).transpose(-1, -2)

        if len(x.shape) == 3:
            return x

        B, C, H, W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return x


class ConvEmbedding(nn.Module):
    """
    Convolutional Patch Embedding proposed by CeiT paper (https://arxiv.org/abs/2103.11816).
    source: https://github.com/vtddggg/Robust-Vision-Transformer/blob/main/robust_models.py
    """

    def __init__(self, in_channels, embed_dim, out_channels, patch_size) -> None:
        """
        params:
            :in_channels: Number of input channels
            :out_channels: Number of output channels
        """
        super(ConvEmbedding, self).__init__()

        self.out_channels = out_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                embed_dim,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(2, 2),
            ),
            nn.BatchNorm2d(embed_dim),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(embed_dim, out_channels, kernel_size=(4, 4), stride=(4, 4)),
        )

    def get_patch_size(self):
        return self.patch_size

    def get_embed_dim(self):
        return self.embed_dim

    def forward(self, x):
        x = (
            self.proj(x)
            .flatten(2)
            .transpose(-1, -2)
            .to(memory_format=torch.contiguous_format)
        )
        return x


class EarlyConv(nn.Module):
    """
    Early Convolutions Help Transformers See Better https://arxiv.org/pdf/2106.14881v2.pdf
    source: https://github.com/Jack-Etheredge/early_convolutions_vit_pytorch/blob/main/vitc/early_convolutions.py

    3x3 conv, stride 1, 5 conv layers per block, 2 blocks
    using this should reduce by one the amount of heads of the transformer
    """

    def __init__(
        self,
        in_ch=3,
        stem_chs=[64, 32, 64],
        out_ch=64,
        with_pos=False,
        strides=[2, 1, 1, 2],
    ):
        super(EarlyConv, self).__init__()
        hidden_ch = out_ch // 2
        self.conv1 = nn.Conv2d(
            in_ch, stem_chs[0], kernel_size=3, stride=strides[0], padding=1, bias=False
        )
        self.norm1 = nn.BatchNorm2d(stem_chs[0])

        self.conv2 = nn.Conv2d(
            stem_chs[0],
            stem_chs[1],
            kernel_size=3,
            stride=strides[1],
            padding=1,
            bias=False,
        )
        self.norm2 = nn.BatchNorm2d(stem_chs[1])

        self.conv3 = nn.Conv2d(
            stem_chs[1],
            stem_chs[2],
            kernel_size=3,
            stride=strides[2],
            padding=1,
            bias=False,
        )
        self.norm3 = nn.BatchNorm2d(stem_chs[2])

        self.conv4 = nn.Conv2d(
            stem_chs[2], out_ch, kernel_size=3, stride=strides[3], padding=1, bias=False
        )
        self.norm4 = nn.BatchNorm2d(out_ch)

        self.act = nn.ReLU(inplace=True)
        self.with_pos = with_pos
        if self.with_pos:
            self.pa_conv = nn.Conv2d(
                out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch
            )
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act(x)

        x = self.conv4(x)
        x = self.norm4(x)
        x = self.act(x)

        x = rearrange(x, "b c h w -> b (h w) c")
        if self.with_pos:
            x = x * self.sigmoid(self.pa_conv(x))
        return x


class BasicStem(nn.Module):
    """
    ResT: An Efficient Transformer for Visual Recognition  https://arxiv.org/pdf/2105.13677.pdf
    -source: https://github.com/wofmanaf/ResT/blob/main/models/rest.py

    basic patch membedding over 3 convs
    """

    def __init__(
        self,
        in_ch=3,
        stem_chs=[64, 32, 64],
        out_ch=64,
        with_pos=False,
        strides=[2, 1, 1, 2],
    ):
        super(BasicStem, self).__init__()
        hidden_ch = out_ch // 2
        self.conv1 = nn.Conv2d(
            in_ch, stem_chs[0], kernel_size=3, stride=strides[0], padding=1, bias=False
        )
        self.norm1 = nn.BatchNorm2d(stem_chs[0])

        self.conv2 = nn.Conv2d(
            stem_chs[0],
            stem_chs[1],
            kernel_size=3,
            stride=strides[1],
            padding=1,
            bias=False,
        )
        self.norm2 = nn.BatchNorm2d(stem_chs[1])

        self.conv3 = nn.Conv2d(
            stem_chs[1],
            stem_chs[2],
            kernel_size=3,
            stride=strides[2],
            padding=1,
            bias=False,
        )
        self.norm3 = nn.BatchNorm2d(stem_chs[2])

        self.conv4 = nn.Conv2d(
            stem_chs[2], out_ch, kernel_size=3, stride=strides[3], padding=1, bias=False
        )
        self.norm4 = nn.BatchNorm2d(out_ch)

        self.act = nn.ReLU(inplace=True)
        self.with_pos = with_pos
        if self.with_pos:
            self.pa_conv = nn.Conv2d(
                out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch
            )
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act(x)

        x = self.conv4(x)
        x = self.norm4(x)
        x = self.act(x)

        if self.with_pos:
            x = x * self.sigmoid(self.pa_conv(x))
        return x


class MedPatchEmbed(nn.Module):
    """ "
    Med patch embedding https://arxiv.org/pdf/2105.13677.pdf
    -source: https://github.com/wofmanaf/ResT/blob/main/models/rest.py
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(MedPatchEmbed, self).__init__()
        self.out_channels = out_channels

        if stride == 2:
            self.avgpool = nn.AvgPool2d(
                (2, 2), stride=2, ceil_mode=True, count_include_pad=False
            )
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, bias=False
            )
            self.norm = nn.BatchNorm2d(out_channels, eps=1e-5)
        elif in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, bias=False
            )
            self.norm = nn.BatchNorm2d(out_channels, eps=1e-5)
        else:
            self.avgpool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        try:
            B, N, C = x.shape
            x = (
                x.transpose(-3, -1)
                .contiguous()
                .view(B, C, int(N**0.5), int(N**0.5))
            )
        except:
            pass
        x = self.norm(self.conv(self.avgpool(x)))
        return x


class GraphPatchEmbed(nn.Module):
    """
    Graph Patch Embedding Module.

    This module applies a Graph Convolutional Network (GCN)
    to the output of the Conv2d layer. This aims to capture spatial relationships between patches more effectively.
    Each patch is treated as a node in a graph, with edges defined based on spatial adjacency.

    Args:
        patch_size (int, optional): Patch size height and width in pixels. Default is 2.
        in_channels (int, optional): Number of input channels. Default is 1.
        embed_dim (int, optional): Embedding dimension for each patch. Default is 192.
        norm_layer (callable, optional): Normalization layer to be applied after convolution. If None, no normalization is applied. Default is None.

    Input Shape:
        - Image tensor of shape (batch_size, in_channels, H, W)

    Output Shape:
        - Output tensor of shape (batch_size, N, embed_dim), where N is the number of patches.
    """

    def __init__(
        self, patch_size=2, in_channels=1, embed_dim=192, norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.norm_layer = norm_layer(embed_dim) if norm_layer else None
        self.gcn = GCNConv(embed_dim, embed_dim)

    def get_embed_dim(self):
        return self.embed_dim

    def get_patch_size(self):
        return self.patch_size

    def forward(self, x):
        print("soy", x.shape)
        x = self.conv(x)
        print("x", x.shape)
        B, C, H, W = x.shape
        # x = x.flatten(2).transpose(-1, -2)  # flatten height and width dimensions
        # N = x.shape[1]

        edge_index = self.get_edge_index(H, W)  # get edge index for GCN
        print("el", x.shape)
        x = self.gcn(x, edge_index)  # forward through GCN
        print("bola", x.shape)
        # x = x.view(B, N, C)  # restore batch dimension

        if self.norm_layer is not None:
            x = self.norm_layer(x)

        return x

    @staticmethod
    def get_edge_index(height, width):
        """
        Creates an adjacency matrix for an image based on its patches.
        For now, it considers the 4 nearest neighbors (up, down, left, right)
        TODO: Consider 8 nearest neighbors (up, down, left, right, and diagonals)
        params:
            :nodes: Number of patches in the image
        returns:
            :edge_index: A tensor containing adjacency information.
                        It has shape (2, E), where E is the number of edges in the graph, linking the nodes/patches.
        """

        nodes = height * width

        edge_index = []
        for i in range(nodes):
            row = i // width
            col = i % width

            if row - 1 >= 0:  # up
                edge_index.append((i, i - width))
            if row + 1 < height:  # down
                edge_index.append((i, i + width))
            if col - 1 >= 0:  # left
                edge_index.append((i, i - 1))
            if col + 1 < width:  # right
                edge_index.append((i, i + 1))

        # 4-diagonal neighbors
        if row - 1 >= 0 and col - 1 >= 0:  # upper left
            edge_index.append((i, i - width - 1))
        if row - 1 >= 0 and col + 1 < width:  # upper right
            edge_index.append((i, i - width + 1))
        if row + 1 < height and col - 1 >= 0:  # lower left
            edge_index.append((i, i + width - 1))
        if row + 1 < height and col + 1 < width:  # lower right
            edge_index.append((i, i + width + 1))

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        return edge_index


class SE(nn.Module):
    """
    Squeeze and excitation block
    """

    def __init__(self, inp, oup, expansion=0.25):
        """
        Args:
            inp: input features dimension.
            oup: output features dimension.
            expansion: expansion ratio.
        """

        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ReduceSize(nn.Module):
    """
    Down-sampling block based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, keep_dim=False):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.GELU(),
            SE(dim, dim),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
        )
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False)
        self.norm2 = norm_layer(dim_out)
        self.norm1 = norm_layer(dim)

    def forward(self, x):
        x = x.contiguous()
        x = self.norm1(x)
        _, C, _ = x.shape
        x = rearrange(x, "b (h w) c -> b c h w", h=int(C**0.5), w=int(C**0.5))
        x = x + self.conv(x)
        x = self.reduction(x)
        x = rearrange(x, "b c h w -> b h w c")
        x = self.norm2(x)
        x = rearrange(x, "b h w c -> b (h w) c")
        return x


class Downsample(nn.Module):
    "Class used to downsample the dimentions of a input to a given output size."

    def __init__(self, in_channels, out_channels, groups):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=2 + 1,
            padding=2 // 2,
            stride=2,
            padding_mode="zeros",
            groups=groups,
        )

    def forward(self, x):
        _, C, _ = x.shape
        x = rearrange(x, "b (h w) c -> b c h w", h=int(C**0.5), w=int(C**0.5)).to(
            memory_format=torch.contiguous_format
        )
        x = self.conv(x)
        x = rearrange(x, "b c h w -> b (h w) c").to(
            memory_format=torch.contiguous_format
        )
        return x
