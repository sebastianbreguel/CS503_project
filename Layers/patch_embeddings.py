from torch import nn

from einops import rearrange


class NaivePatchEmbed(nn.Module):
    """
    Basic Patch Embedding Module. Same as in the transformers graded notebook.
    """

    def __init__(self, patch_size=2, in_channels=1, embed_dim=192, norm_layer=None):
        """
        Image to Patch Embedding.

        params:
            :patch_size: Patch size height and width in pixels
            :in_channels: Number of input channels
            :embed_dim: Token dimension
        """
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
    """

    def __init__(self, in_channels, embed_dim, out_channels, patch_size):
        """
        params:
            :in_channels: Number of input channels
            :out_channels: Number of output channels
        """
        super().__init__()

        self.out_channels = out_channels
        self.patch_size = patch_size

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
        return self.out_channels

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(-1, -2)
        return x


class Image2Tokens(nn.Module):
    def __init__(self, in_chans=3, out_chans=64, kernel_size=7, stride=2):
        super(Image2Tokens, self).__init__()
        self.conv = nn.Conv2d(
            in_chans,
            out_chans,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_chans)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.maxpool(x)
        return x


class conv_head_pooling(nn.Module):
    def __init__(self, in_feature, out_feature, stride, padding_mode="zeros"):
        super(conv_head_pooling, self).__init__()

        self.conv = nn.Conv2d(
            in_feature,
            out_feature,
            kernel_size=stride + 1,
            padding=stride // 2,
            stride=stride,
            padding_mode=padding_mode,
            groups=in_feature,
        )

    def forward(self, x):
        x = self.conv(x)

        return x
