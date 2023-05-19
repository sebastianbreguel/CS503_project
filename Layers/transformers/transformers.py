import torch
import torch.nn as nn

from .blocks import Block, CustomBlock, Parallel_blocks, RobustBlock


class Transformer(nn.Module):
    """
    Initial transformer class https://arxiv.org/abs/1706.03762
    """

    def __init__(
        self, dim, depth, num_heads=8, mlp_ratio=4.0, drop_rate=0.0, masked_block=None
    ) -> None:
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
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
        super().__init__()
        self.layers = nn.ModuleList([])
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
        super().__init__()
        self.layers = nn.ModuleList([])
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


class RVTransformer(nn.Module):
    """
    Transformers Blocks for the RVT model https://github.com/vtddggg/Robust-Vision-Transformer/blob/main/robust_models.py

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
        super().__init__()
        self.layers = nn.ModuleList([])
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
