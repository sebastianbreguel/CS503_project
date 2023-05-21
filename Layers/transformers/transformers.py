import torch
import torch.nn as nn

from .blocks import Block, CustomBlock, ECBlock, LTBlock, Parallel_blocks, RobustBlock


class Transformer(nn.Module):
    """
    Initial transformer class https://arxiv.org/abs/1706.03762
    Same as in the transformers graded notebook.
    """

    def __init__(
        self, dim, depth, num_heads=8, mlp_ratio=4.0, drop_rate=0.0, masked_block=None
    ) -> None:
        super(Transformer, self).__init__()
        self.depth = depth
        self.blocks = nn.ModuleList()

        for _ in range(depth):
            self.blocks.append(
                Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate)
            )

    def forward(self, x) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class ParallelTransformers(nn.Module):
    """Parallel transformer (N parallel attention followed by N parallel MLP)
    Based on:
      `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795

    """

    def __init__(
        self, dim, depth, num_heads=8, mlp_ratio=4.0, drop_rate=0.0, masked_block=None
    ) -> None:
        super(ParallelTransformers, self).__init__()
        self.parallel_blocks = 3
        self.depth = depth // self.parallel_blocks
        self.unique = depth % self.parallel_blocks
        self.blocks = nn.ModuleList()

        for _ in range(depth):
            self.blocks.append(
                Parallel_blocks(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    num_parallel=self.parallel_blocks,
                )
            )

        for _ in range(self.unique):
            self.blocks.append(
                Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate)
            )

    def forward(self, x) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class CustomTransformer(nn.Module):

    """
    Class to use custom blocks in the transformer modifications
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.0,
        masked_block=None,
        block=CustomBlock,
    ) -> None:
        super(CustomTransformer, self).__init__()
        self.depth = depth
        self.blocks = nn.ModuleList()

        for _ in range(depth):
            self.blocks.append(
                block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate)
            )

    def forward(self, x) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


####################################
# Especific transformer architectures
####################################


class RVTransformer(nn.Module):
    """
    Transformers Blocks for the RVT model https://arxiv.org/pdf/2105.07926.pdf
    - source: https://github.com/vtddggg/Robust-Vision-Transformer/tree/main
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.0,
        masked_block=None,
        block=RobustBlock,
    ) -> None:
        super(RVTransformer, self).__init__()
        self.depth = depth
        self.blocks = nn.ModuleList()
        self.pooling = nn.ModuleList()

        for _ in range(depth):
            self.blocks.append(
                Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate)
            )
            self.pooling.append(
                nn.Conv2d(
                    dim,
                    dim,
                    kernel_size=2 + 1,
                    padding=2 // 2,
                    stride=2,
                    padding_mode="zeros",
                    groups=dim,
                )
            )

    def forward(self, x) -> torch.Tensor:
        for state in range(self.depth):
            x = self.blocks[state](x)
            x = self.pooling[state](x)

        return x


class MedVitTransformer(nn.Module):
    """
    Transformers "Phase/Blocks" for the MedViT model https://arxiv.org/abs/2302.09462

    -source: https://github.com/Omid-Nejati/MedViT/tree/main
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.0,
        ecb=None,
        ltb=None,
        strides=None,
        sr_ratios=None,
    ) -> None:
        super(MedVitTransformer, self).__init__()

        self.num_ecb = ecb
        self.num_ltb = ltb
        self.strides = strides
        self.sr_ratios = sr_ratios

        self.blocks = []

        for _ in range(ecb):
            self.blocks.append(
                ECBlock(
                    in_channels=dim,
                    out_channels=dim,
                    drop=drop_rate,
                    mlp_ratio=mlp_ratio,
                    num_heads=num_heads,
                    stride=strides[_],
                )
            )

        for _ in range(ltb):
            self.blocks.append(
                LTBlock(
                    in_channels=dim,
                    out_channels=dim,
                    drop=drop_rate,
                    sr_ratio=sr_ratios[_],
                    stride=strides[_ + ecb],
                    mlp_ratio=mlp_ratio,
                    num_heads=num_heads,
                )
            )

        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x
