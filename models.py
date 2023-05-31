import ast

import torch
import torch.nn as nn

from einops.layers.torch import Rearrange
from Layers import (
    ECBlock,
    LTBlock,
    Block,
    EarlyConv,
    BasicStem,
    ConvEmbedding,
    CustomTransformer,
    Downsample,
    GraphPatchEmbed,
    trunc_normal_,
    LayerNorm,
    MedVitTransformer,
    MedPatchEmbed,
    Model1ParallelBlock,
    NaivePatchEmbed,
    RVTransformer,
    RobustBlock,
    ReduceSize,
    ParallelTransformers,
    SineCosinePosEmbedding,
    Transformer,
)


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
        num_classes=101,
        strides=[1, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        num_heads=32,
        mix_block_ratio=0.75,
    ):
        super(MedViT, self).__init__()

        self.stem = BasicStem(3)

        self.stage_out_channels = [
            [96] * (depths[0]),
            [128] * (depths[1] - 1) + [192],
            [256, 384, 384] * (depths[2] // 3),
            [384] * (depths[3] - 1) + [512],
        ]

        # Next Hybrid Strategy
        self.stage_block_types = [
            [ECBlock] * depths[0],
            [ECBlock] * (depths[1] - 1) + [LTBlock],
            [ECBlock, ECBlock, LTBlock] * (depths[2] // 3),
            [ECBlock] * (depths[3] - 1) + [LTBlock],
        ]
        input_channel = 64
        features = []
        idx = 0
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]  # stochastic depth decay rule
        for stage_id in range(len(depths)):
            numrepeat = depths[stage_id]
            output_channels = self.stage_out_channels[stage_id]
            block_types = self.stage_block_types[stage_id]
            for block_id in range(numrepeat):
                if strides[stage_id] == 2 and block_id == 0:
                    stride = 2
                else:
                    stride = 1
                output_channel = output_channels[block_id]
                block_type = block_types[block_id]
                if block_type is ECBlock:
                    layer = ECBlock(
                        in_channels=input_channel,
                        out_channels=output_channel,
                        drop=drop_rate,
                        stride=stride,
                        path_dropout=dpr[idx + block_id],
                    )
                    features.append(layer)
                elif block_type is LTBlock:
                    layer = LTBlock(
                        in_channels=input_channel,
                        out_channels=output_channel,
                        sr_ratio=sr_ratios[stage_id],
                        drop=drop_rate,
                        num_heads=num_heads,
                        stride=stride,
                    )
                    features.append(layer)
                input_channel = output_channel
            idx += numrepeat
        self.features = nn.Sequential(*features)

        self.features = nn.Sequential(*self.features)

        self.norm = nn.BatchNorm2d(input_channel, eps=1e-6)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj_head = nn.Sequential(
            nn.Linear(input_channel, num_classes),
        )
        print(num_classes)
        self.stage_out_idx = [sum(depths[: idx + 1]) - 1 for idx in range(len(depths))]
        print("initialize_weights...")
        self._initialize_weights()

    def _initialize_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        for layer in self.features:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.proj_head(x)
        return x


class Testion(nn.Module):
    def __init__(self, depth=[4, 2], num_heads=4, mlp_ratio=4.0, drop_rate=0.0, patch_embedding="default", positional_encoding=False, img_size=(224, 224), num_classes=10, head_bias=False, **kwargs):
        super(Testion, self).__init__()
        self.num_classes = num_classes
        self.head_bias = head_bias
        self.patch_embedding = EarlyConv(3, stem_chs=[16, 32, 64], out_ch=128, strides=[2, 2, 2, 2])

        initial_size = img_size[0] // 16

        self.blocks = nn.ModuleList()
        self.depth = depth
        dpr = [x.item() for x in torch.linspace(0, drop_rate, sum(depth))]  # stochastic depth decay rule
        dims = [128, 256]
        groups = [64]
        self.downsamples = nn.ModuleList()
        self.parallels = nn.ModuleList()
        final_attention = [True, True]

        for _ in range(self.depth[0]):
            drop = dpr[_]
            self.blocks.append(
                RobustBlock(
                    dim=dims[0],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    size=initial_size,
                )
            )

        for stage in range(len(self.depth) - 1):
            aux_size = initial_size // ((stage + 1) * 2)
            drop = dpr[self.depth[0] + stage : self.depth[0] + stage + self.depth[stage + 1]]
            self.downsamples.append(
                ReduceSize(
                    dims[stage],
                )
            )

            self.parallels.append(
                ParallelTransformers(
                    dim=dims[stage + 1],
                    depth=self.depth[stage + 1],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop,
                    size=aux_size,
                    final=final_attention[stage],
                )
            )

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.gap = nn.Sequential(
            # REARANGE
            Rearrange(
                "batch (height width) channels  -> batch  channels height width",
                height=aux_size,
                width=aux_size,
            ),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Classifier head
        self.head = nn.Linear(dims[-1], num_classes, bias=head_bias)
        self.head_drop = nn.Dropout(dpr[-1])
        print("init weights...")
        self._initialize_weights()

    def _initialize_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.patch_embedding(x)
        for state in range(self.depth[0]):
            x = self.blocks[state](x)

        for stage in range(len(self.depth) - 1):
            x = self.downsamples[stage](x)
            x = self.parallels[stage](x)

        x = self.norm(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        x = self.head_drop(x)

        return x


class MedViT_S(nn.Module):
    """
    MedViT: A Robust Vision Transformer for Generalized Medical Image Classification https://arxiv.org/abs/2302.09462
    -source: https://github.com/Omid-Nejati/MedViT/blob/main/MedViT.py
    """

    def __init__(
        self,
        stem_chs,
        depths,
        drop_rate,
        num_classes=101,
        strides=[1, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        num_heads=32,
        mix_block_ratio=0.75,
    ):
        super(MedViT_S, self).__init__()

        self.stem = BasicStem(3)

        self.stage_out_channels = [
            [96] * (depths[0]),
            [128] * (depths[1] - 1) + [192],
            [256, 384] * (depths[2] // 2),
            [384] * (depths[3] - 1) + [512],
        ]

        # Next Hybrid Strategy
        self.stage_block_types = [
            [ECBlock] * depths[0],
            [ECBlock] * (depths[1] - 1) + [LTBlock],
            [ECBlock, LTBlock] * (depths[2] // 2),
            [ECBlock] * (depths[3] - 1) + [LTBlock],
        ]

        input_channel = 64
        features = []
        idx = 0
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]  # stochastic depth decay rule
        for stage_id in range(len(depths)):
            numrepeat = depths[stage_id]
            output_channels = self.stage_out_channels[stage_id]
            block_types = self.stage_block_types[stage_id]
            for block_id in range(numrepeat):
                if strides[stage_id] == 2 and block_id == 0:
                    stride = 2
                else:
                    stride = 1
                output_channel = output_channels[block_id]
                block_type = block_types[block_id]
                if block_type is ECBlock:
                    layer = ECBlock(
                        in_channels=input_channel,
                        out_channels=output_channel,
                        drop=drop_rate,
                        stride=stride,
                        path_dropout=dpr[idx + block_id],
                    )
                    features.append(layer)
                elif block_type is LTBlock:
                    layer = LTBlock(
                        in_channels=input_channel,
                        out_channels=output_channel,
                        sr_ratio=sr_ratios[stage_id],
                        drop=drop_rate,
                        num_heads=num_heads,
                        stride=stride,
                    )
                    features.append(layer)
                input_channel = output_channel
            idx += numrepeat
        self.features = nn.Sequential(*features)

        self.features = nn.Sequential(*self.features)

        self.norm = nn.BatchNorm2d(input_channel, eps=1e-6)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj_head = nn.Sequential(
            nn.Linear(input_channel, num_classes),
        )

        self.stage_out_idx = [sum(depths[: idx + 1]) - 1 for idx in range(len(depths))]
        print("initialize_weights...")
        self._initialize_weights()

    def _initialize_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        for layer in self.features:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.proj_head(x)
        return x
