from collections import OrderedDict

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import math
import warnings

######
# Normalization methods
#####


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6) -> None:
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        return self.norm(x)


class RMSNorm(nn.Module):
    """
    "Root Mean Square Layer Normalization (RMSNorm)" -> https://arxiv.org/pdf/1910.07467.pdf
    """

    def __init__(self, dim, eps=1e-8) -> None:
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.dim = dim
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x * self.dim ** (-1 / 2)

        x_normed = x / (rms_x + self.eps)

        return self.gamma * x_normed


class DeepNormalize(nn.Module):
    """
    DeepNormalize: An Efficient Normalization Method for Deep Learning
    """

    def __init__(self, alpha) -> None:
        super(DeepNormalize, self).__init__()
        self.alpha = alpha
        self.layerNorm = nn.LayerNorm()

    def forward(self, layer, x):
        return self.layerNorm(self.alpha * x + layer(x))


#######################################################
# Activation functions
#######################################################


class SquaredRelu(nn.Module):
    def __init__(self) -> None:
        super(SquaredRelu, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.relu(x)
        return x * x


class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        return x * self.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self) -> None:
        super(SwiGLU, self).__init__()
        self.swish = nn.Swish()

    def forward(self, x1, x2):
        return self.swish(x1) * x2


class GeGlu(nn.Module):
    def __init__(self) -> None:
        super(GeGlu, self).__init__()
        self.gelu = nn.GELU()

    def forward(self, x1, x2):
        return self.gelu(x1) * x2


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


###############################################################


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    source: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    - source: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class PatchDropout(torch.nn.Module):
    """
    Implements PatchDropout: https://arxiv.org/abs/2208.07220
    source: https://github.com/yueliukth/PatchDropout/tree/main
    """

    def __init__(self, keep_rate=0.5, sampling="uniform", token_shuffling=False):
        super(PatchDropout, self).__init__()
        assert 0 < keep_rate <= 1, "The keep_rate must be in (0,1]"

        self.keep_rate = keep_rate
        self.sampling = sampling
        self.token_shuffling = token_shuffling

    def forward(self, x, force_drop=False):
        """
        If force drop is true it will drop the tokens also during inference.
        """
        if not self.training and not force_drop:
            return x
        if self.keep_rate == 1:
            return x

        # batch, length, dim
        N, L, D = x.shape

        # making cls mask (assumes that CLS is always the 1st element)
        cls_mask = torch.zeros(N, 1, dtype=torch.int64, device=x.device)
        # generating patch mask
        patch_mask = self.get_mask(x)

        # cat cls and patch mask
        patch_mask = torch.hstack([cls_mask, patch_mask])
        # gather tokens
        x = torch.gather(x, dim=1, index=patch_mask.unsqueeze(-1).repeat(1, 1, D))

        return x

    def get_mask(self, x):
        if self.sampling == "uniform":
            return self.uniform_mask(x)
        else:
            return NotImplementedError(f"PatchDropout does ot support {self.sampling} sampling")

    def uniform_mask(self, x):
        """
        Returns an id-mask using uniform sampling
        """
        N, L, D = x.shape
        _L = L - 1  # patch lenght (without CLS)

        keep = int(_L * self.keep_rate)
        patch_mask = torch.rand(N, _L, device=x.device)
        patch_mask = torch.argsort(patch_mask, dim=1) + 1
        patch_mask = patch_mask[:, :keep]
        if not self.token_shuffling:
            patch_mask = patch_mask.sort(1)[0]
        return patch_mask


def _build_projection(dim_in, dim_out, kernel_size, padding, stride, method):
    if method == "dw_bn":
        proj = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            dim_in,
                            dim_out,
                            kernel_size=kernel_size,
                            padding=padding,
                            stride=stride,
                            bias=False,
                            groups=dim_in,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(dim_out)),
                    ("rearrage", Rearrange("b c h w -> b (h w) c")),
                ]
            )
        )
    elif method == "avg":
        proj = nn.Sequential(
            OrderedDict(
                [
                    (
                        "avg",
                        nn.AvgPool2d(
                            kernel_size=kernel_size,
                            padding=padding,
                            stride=stride,
                            ceil_mode=True,
                        ),
                    ),
                    ("rearrage", Rearrange("b c h w -> b (h w) c")),
                ]
            )
        )
    elif method == "linear":
        proj = None
    else:
        raise ValueError("Unknown method ({})".format(method))

    return proj


# for the MedVit transformer


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor

    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


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


class SELayer(nn.Module):
    """
    SELayer  https://arxiv.org/abs/2302.09462
    -source: https://github.com/Omid-Nejati/MedViT/blob/main/MedViT.py
    """

    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            h_sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# TODO: check this part
def moex(x, swap_index, norm_type, epsilon=1e-5, positive_only=False):
    """
    Moment exchanger patch https://arxiv.org/abs/2302.09462
    -source: https://github.com/Omid-Nejati/MedViT/blob/main/MedViT.py
    """
    dtype = x.dtype
    x = x.float()

    B, C, L = x.shape
    if norm_type == "bn":
        norm_dims = [0, 2, 3]
    elif norm_type == "in":
        norm_dims = [2, 3]
    elif norm_type == "ln":
        norm_dims = [1, 2, 3]
    elif norm_type == "pono":
        norm_dims = [1]
    elif norm_type.startswith("gn"):
        if norm_type.startswith("gn-d"):
            # gn-d4 means GN where each group has 4 dims
            G_dim = int(norm_type[4:])
            G = C // G_dim
        else:
            # gn4 means GN with 4 groups
            G = int(norm_type[2:])
            G_dim = C // G
        x = x.view(B, G, G_dim, H, W)
        norm_dims = [2, 3, 4]
    elif norm_type.startswith("gpono"):
        if norm_type.startswith("gpono-d"):
            # gpono-d4 means GPONO where each group has 4 dims
            G_dim = int(norm_type[len("gpono-d") :])
            G = C // G_dim
        else:
            # gpono4 means GPONO with 4 groups
            G = int(norm_type[len("gpono") :])
            G_dim = C // G
        x = x.view(B, G, G_dim, H, W)
        norm_dims = [2]
    else:
        raise NotImplementedError(f"norm_type={norm_type}")

    if positive_only:
        x_pos = F.relu(x)
        s1 = x_pos.sum(dim=norm_dims, keepdim=True)
        s2 = x_pos.pow(2).sum(dim=norm_dims, keepdim=True)
        count = x_pos.gt(0).sum(dim=norm_dims, keepdim=True)
        count[count == 0] = 1  # deal with 0/0
        mean = s1 / count
        var = s2 / count - mean.pow(2)
        std = var.add(epsilon).sqrt()
    else:
        mean = x.mean(dim=norm_dims, keepdim=True)
        std = x.var(dim=norm_dims, keepdim=True).add(epsilon).sqrt()
    swap_mean = mean[swap_index]
    swap_std = std[swap_index]
    # output = (x - mean) / std * swap_std + swap_mean
    # equvalent but for efficient
    scale = swap_std / std
    shift = swap_mean - mean * scale
    output = x * scale + shift
    return output.view(B, C, L).to(dtype)


def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. " "The distribution of values may be incorrect.", stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)
