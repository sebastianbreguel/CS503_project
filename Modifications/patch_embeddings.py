from torch import nn


class PatchEmbed(nn.Module):
    '''
    Basic Patch Embedding Module. Same as in the transformers graded notebook.
    '''
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

        self.conv = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size, bias=False)

    def forward(self, x):
        '''        
        params:
            :x: Input of shape [B C H W]. B = batch size, C = number of channels, H = image height, W = image width
        returns:
            Output of shape [B N C].
        '''
        return self.conv(x).flatten(2).transpose(1, 2)


class ConvEmbedding(nn.Module):
    '''
    Convolutional Patch Embedding proposed by CeiT paper (https://arxiv.org/abs/2103.11816).
    '''
    def __init__(self, in_channels, out_channels, patch_size, stride, padding):
        '''
        params:
            :in_channels: Number of input channels
            :out_channels: Number of output channels
        '''
        super().__init__()

        self.out_channels = out_channels

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(7, 7), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(32, out_channels, kernel_size=(4, 4), stride=(4, 4)),
        )

    def forward(self, x):
        x = self.proj(x)
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
