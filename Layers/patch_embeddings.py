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

    def __init__(self, patch_size=2, in_channels=1, embed_dim=192, norm_layer=None) -> None:
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
        x = self.proj(x).flatten(2).transpose(-1, -2).to(memory_format=torch.contiguous_format)
        return x


class EarlyConv(nn.Module):
    """
    Early Convolutions Help Transformers See Better https://arxiv.org/pdf/2106.14881v2.pdf
    source: https://github.com/Jack-Etheredge/early_convolutions_vit_pytorch/blob/main/vitc/early_convolutions.py

    3x3 conv, stride 1, 5 conv layers per block, 2 blocks
    using this should reduce by one the amount of heads of the transformer
    """

    def __init__(self, channels, dim, emb_dropout=0.0) -> None:
        super(EarlyConv, self).__init__()
        n_filter_list = (
            channels,
            48,
            96,
            192,
            384,
        )  # hardcoding for now because that's what the paper used

        self.conv_layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=n_filter_list[i],
                        out_channels=n_filter_list[i + 1],
                        kernel_size=3,  # hardcoding for now because that's what the paper used
                        stride=2,  # hardcoding for now because that's what the paper used
                        padding=1,
                    ),  # hardcoding for now because that's what the paper used
                )
                for i in range(len(n_filter_list) - 1)
            ]
        )

        self.conv_layers.add_module(
            "conv_1x1",
            nn.Conv2d(
                in_channels=n_filter_list[-1],
                out_channels=dim,
                stride=1,  # hardcoding for now because that's what the paper used
                kernel_size=1,  # hardcoding for now because that's what the paper used
                padding=0,
            ),
        )  # hardcoding for now because that's what the paper used
        self.conv_layers.add_module(
            "flatten image",
            Rearrange("batch channels height width -> batch (height width) channels"),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, n_filter_list[-1] + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x):
        x = self.conv_layers(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)


class BasicStem(nn.Module):
    """
    ResT: An Efficient Transformer for Visual Recognition  https://arxiv.org/pdf/2105.13677.pdf
    -source: https://github.com/wofmanaf/ResT/blob/main/models/rest.py

    basic patch membedding over 3 convs
    """

    def __init__(self, in_ch=3, out_ch=64, with_pos=False):
        super(BasicStem, self).__init__()
        hidden_ch = out_ch // 2
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(hidden_ch)
        self.conv3 = nn.Conv2d(hidden_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)

        self.act = nn.ReLU(inplace=True)
        self.with_pos = with_pos
        if self.with_pos:
            self.pa_conv = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)

        x = self.conv3(x)
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

        if stride == 2:
            self.avgpool = nn.AvgPool2d((2, 2), stride=2, ceil_mode=True, count_include_pad=False)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = nn.BatchNorm2d(out_channels, eps=1e-5)
        elif in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = nn.BatchNorm2d(out_channels, eps=1e-5)
        else:
            self.avgpool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.conv(self.avgpool(x)))


# TODO Check if we will use this -> finish implementation
class PixelEmbed(nn.Module):
    """Image to Pixel Embedding
    - source: https://github.com/Omid-Nejati/Locality-iN-Locality/blob/main/models/tnt.py
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, in_dim=48, stride=4):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.img_size = img_size
        self.num_patches = num_patches
        self.in_dim = in_dim
        new_patch_size = math.ceil(patch_size / stride)
        self.new_patch_size = new_patch_size

        self.proj = nn.Conv2d(in_chans, self.in_dim, kernel_size=7, padding=3, stride=stride)
        self.unfold = nn.Unfold(kernel_size=new_patch_size, stride=new_patch_size)
        self.pixel_pos = nn.Parameter(torch.zeros(1, in_dim, new_patch_size, new_patch_size))

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x)
        x = self.unfold(x)
        x = x.transpose(1, 2).reshape(B * self.num_patches, self.in_dim, self.new_patch_size, self.new_patch_size)
        x = x + self.pixel_pos
        x = x.reshape(B * self.num_patches, self.in_dim, -1).transpose(1, 2)
        return x


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        in_dim=48,
        drop=0.0,
        norm_layer=nn.LayerNorm,
        first_stride=4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.pixel_embed = PixelEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            in_dim=in_dim,
            stride=first_stride,
        )

        num_patches = self.pixel_embed.num_patches
        self.num_patches = num_patches
        new_patch_size = self.pixel_embed.new_patch_size
        num_pixel = new_patch_size**2

        self.norm1_proj = norm_layer(num_pixel * in_dim)
        self.proj = nn.Linear(num_pixel * in_dim, embed_dim)
        self.norm2_proj = norm_layer(embed_dim)

        self.patch_pos = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.pos_drop = nn.Dropout(drop)

    def forward(self, x):
        B, C, H, W = x.shape

        pixel_embed = self.pixel_embed(x)

        patch_embed = self.norm2_proj(self.proj(self.norm1_proj(pixel_embed.reshape(B, self.num_patches, -1))))
        patch_embed = patch_embed + self.patch_pos
        patch_embed = self.pos_drop(patch_embed)
        return pixel_embed, patch_embed


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

    def __init__(self, patch_size=2, in_channels=1, embed_dim=192, norm_layer=None):
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
        x = self.conv(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(-1, -2)  # flatten height and width dimensions
        N = x.shape[1]

        x = x.contiguous().view(B * N, C)  # flatten batch and patch dimensions

        edge_index = self.get_edge_index(H, W)  # get edge index for GCN
        x = self.gcn(x, edge_index)  # forward through GCN
        x = x.view(B, N, C)  # restore batch dimension

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