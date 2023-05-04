from models import ViT
import torch
from utils import get_dataset, get_device, get_loss, train_model


if __name__ == "__main__":

    device = get_device()

    vit = ViT(
        img_size=14,
        patch_size=2,
        in_channels=1,
        embed_dim=192,
        num_classes=10,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        drop=0.15,
    ).to(device)

    optimizer = torch.optim.AdamW(vit.parameters())
    num_parameters = sum([p.numel() for p in vit.parameters()])
    print(f"Number of parameters: {num_parameters:,}")

    num_epochs = 2

    loader_train, loader_val, loader_test = get_dataset("MNIST")

    loss = get_loss("cross entropy")

    train_model(vit, optimizer, loader_train, loader_val, num_epochs, loss, device)
