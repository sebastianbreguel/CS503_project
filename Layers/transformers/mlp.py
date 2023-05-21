import torch
import torch.nn as nn


class Mlp(nn.Module):
    """
    Base two layer MLP. From the transformers notebook.

    trad
    """

    def __init__(
        self, dim, activation_function=nn.GELU, dropout=0.0, mlp_ratio=4.0
    ) -> None:
        """
        params:
            :dim: Dimensionality of each token
            :activation_function: Activation function to use
            :dropout: Dropout rate
            :mlp_ratio: MLP hidden dimensionality multiplier
        """
        super(Mlp, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(mlp_ratio * dim)),
            activation_function(),
            nn.Dropout(dropout),
            nn.Linear(int(mlp_ratio * dim), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x) -> torch.Tensor:
        return self.mlp(x)  # returns output of the same dimension as the input


class RobustMlp(nn.Module):
    """
    https://github.com/vtddggg/Robust-Vision-Transformer/blob/main/robust_models.py
    """

    def __init__(
        self, dim, activation_function=nn.GELU, dropout=0.0, mlp_ratio=4.0
    ) -> None:
        """
        params:
            :dim: Dimensionality of each token
            :activation_function: Activation function to use
            :dropout: Dropout rate
            :mlp_ratio: MLP hidden dimensionality multiplier
        """
        super(RobustMlp, self).__init__()
        self.hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Conv2d(dim, self.hidden_features, 1)
        self.bn1 = nn.BatchNorm2d(self.hidden_features)
        self.dwconv = nn.Conv2d(
            self.hidden_features,
            self.hidden_features,
            3,
            padding=1,
            groups=self.hidden_features,
        )
        self.bn2 = nn.BatchNorm2d(self.hidden_features)
        self.act = activation_function()
        self.fc2 = nn.Conv2d(self.hidden_features, dim, 1)
        self.bn3 = nn.BatchNorm2d(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x) -> torch.Tensor:
        B, N, C = x.shape
        x = (
            x.reshape(B, int(N**0.5), int(N**0.5), C)
            .permute(0, 3, 1, 2)
            .to(memory_format=torch.contiguous_format)
        )
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dwconv(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.bn3(x)
        x = self.drop(x)
        x = x.permute(0, 2, 3, 1).reshape(B, -1, C)
        return x
