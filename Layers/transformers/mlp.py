import torch.nn as nn


class Mlp(nn.Module):
    """
    Base two layer MLP. From the transformers notebook.

    trad
    """

    def __init__(self, dim, activation_function=nn.GELU, dropout=0.0, mlp_ratio=4.0):
        """
        params:
            :dim: Dimensionality of each token
            :activation_function: Activation function to use
            :dropout: Dropout rate
            :mlp_ratio: MLP hidden dimensionality multiplier
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(mlp_ratio * dim)),
            activation_function(),
            nn.Dropout(dropout),
            nn.Linear(int(mlp_ratio * dim), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mlp(x)  # returns output of the same dimension as the input
