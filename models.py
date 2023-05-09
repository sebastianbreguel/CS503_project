from base_models import build_2d_sincos_posemb, Transformer
from Modifications import PatchEmbed
import torch
import torch.nn as nn
import math
from Modifications import conv_embedding, conv_head_pooling


class ViT(nn.Module):
    def __init__(
        self,
        img_size=14,
        patch_size=2,
        in_channels=3,
        embed_dim=1280,
        num_classes=100,
        depth=32,
        num_heads=16,
        mlp_ratio=4.0,
        head_bias=False,
        drop=0.1,
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

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(
            build_2d_sincos_posemb(
                img_size // patch_size, img_size // patch_size, embed_dim=embed_dim
            ),
            requires_grad=False,
        )

        self.transformer = Transformer(
            dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes, bias=head_bias)
        )

    def forward(self, x):
        proj = self.patch_embed(x)
        proj = proj + self.pos_embed

        proj = self.transformer(proj)

        pooled = proj.mean(dim=1)
        logits = self.head(pooled)

        return logits


class PoolingTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        stride,
        base_dims,
        depth,
        heads,
        mlp_ratio,
        num_classes=1000,
        in_chans=3,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_mask=False,
        masked_block=None,
    ):
        super(PoolingTransformer, self).__init__()

        total_block = sum(depth)
        padding = 0
        block_idx = 0

        width = math.floor((image_size / stride))

        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes

        self.patch_size = patch_size
        self.patch_embed = conv_embedding(
            in_chans, base_dims[0] * heads[0], patch_size, stride, padding
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.transformers = nn.ModuleList([])
        self.pools = nn.ModuleList([])

        for stage in range(len(depth)):
            drop_path_prob = [
                drop_path_rate * i / total_block
                for i in range(block_idx, block_idx + depth[stage])
            ]
            block_idx += depth[stage]
            if stage == 0:
                self.transformers.append(
                    Transformer(
                        base_dims[stage],
                        depth[stage],
                        heads[stage],
                        mlp_ratio,
                        drop_rate,
                        attn_drop_rate,
                        drop_path_prob,
                        use_mask=use_mask,
                        masked_block=masked_block,
                    )
                )
            else:
                self.transformers.append(
                    Transformer(
                        base_dims[stage],
                        depth[stage],
                        heads[stage],
                        mlp_ratio,
                        drop_rate,
                        attn_drop_rate,
                        drop_path_prob,
                    )
                )
            if stage < len(heads) - 1:
                self.pools.append(
                    conv_head_pooling(
                        base_dims[stage] * heads[stage],
                        base_dims[stage + 1] * heads[stage + 1],
                        stride=2,
                    )
                )
        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], eps=1e-6)
        self.embed_dim = base_dims[-1] * heads[-1]
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier head
        if num_classes > 0:
            self.head = nn.Linear(base_dims[-1] * heads[-1], num_classes)
        else:
            self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        if num_classes > 0:
            self.head = nn.Linear(self.embed_dim, num_classes)
        else:
            self.head = nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for stage in range(len(self.pools)):
            x = self.transformers[stage](x)
            x = self.pools[stage](x)
        x = self.transformers[-1](x)
        cls_features = self.norm(self.gap(x).squeeze())

        return cls_features

    def forward(self, x):
        cls_features = self.forward_features(x)
        output = self.head(cls_features)
        return output
