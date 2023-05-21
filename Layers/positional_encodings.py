import torch
import torch.nn as nn


class SineCosinePosEmbedding(nn.Module):

    """Sine-cosine positional embeddings from MoCo-v3
    Source: https://github.com/facebookresearch/moco-v3/blob/main/vits.py
    Returns positional embedding of shape [B, N, D]
    """

    def __init__(
        self,
        height,
        width,
        embed_dim: int = 1024,
        temperature: float = 10000.0,
        requires_grad: bool = False,
    ) -> None:
        super(SineCosinePosEmbedding, self).__init__()
        self.height = height
        self.width = width
        self.embed_dim = embed_dim
        self.temperature = temperature

        grid_w = torch.arange(width, dtype=torch.float32)
        grid_h = torch.arange(height, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        assert (
            embed_dim % 4 == 0
        ), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)
        out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
        out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
        self.pos_emb = nn.Parameter(
            torch.cat(
                [
                    torch.sin(out_w),
                    torch.cos(out_w),
                    torch.sin(out_h),
                    torch.cos(out_h),
                ],
                dim=1,
            )[None, :, :],
            requires_grad=requires_grad,
        )

    def forward(self, x):
        batch_size = x.size(0)
        pos_emb = self.pos_emb.repeat(batch_size, 1, 1)
        return pos_emb


class RelativePos(nn.Module):
    def __init__(self, dim, num_heads, head_dim) -> None:
        super(RelativePos, self).__init__()
        self.head_dim = head_dim
        self.max_length = 1028
        self.Er = torch.randn(
            [num_heads, self.max_length, self.head_dim],
            requires_grad=True,
        )

    def forward(self, q, N, transpose=True):
        embedding_start = self.max_length - N
        Er = self.Er[:, embedding_start:, :].unsqueeze(0)
        if transpose:
            Er = Er.transpose(-1, -2)
        QEr = torch.matmul(q, Er.to(q.device))
        return QEr
