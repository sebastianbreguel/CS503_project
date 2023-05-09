import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST
import torch.nn.functional as F


def get_dataset(name):
    if name == "MNIST":
        image_size = 14
        batch_size = 64
        num_workers = 8

        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((image_size, image_size)),
                torchvision.transforms.ToTensor(),
            ]
        )

        dataset_train_val = MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        dataset_train, dataset_val = torch.utils.data.random_split(
            dataset_train_val, [50_000, 10_000]
        )
        dataset_test = MNIST(
            root="./data", train=False, download=True, transform=transform
        )

    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
    )
    loader_val = DataLoader(
        dataset_val, batch_size=batch_size, num_workers=num_workers, drop_last=False
    )
    loader_test = DataLoader(
        dataset_test, batch_size=batch_size, num_workers=num_workers, drop_last=False
    )

    return loader_train, loader_val, loader_test


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Selected device:", device)

    return device


def get_loss(name):
    if name == "cross entropy":
        loss = F.cross_entropy

    return loss


def train_model(
    model, optimizer, loader_train, loader_val, num_epochs, loss_function, device="cpu"
):
    train_losses = []
    val_losses = []

    for _ in range(num_epochs):
        # Train loop
        model.train()
        epoch_loss_train = 0
        for imgs, cls_idxs in loader_train:
            inputs, targets = imgs.to(device), cls_idxs.to(device)
            logits = model(inputs)
            loss = loss_function(logits, targets)

            model.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss_train += loss.item()

        epoch_loss_train /= len(loader_train)
        train_losses.append(epoch_loss_train)

        # Validation loop
        model.eval()
        epoch_loss_val = 0
        for imgs, cls_idxs in loader_val:
            inputs, targets = imgs.to(device), cls_idxs.to(device)
            logits = model(inputs)
            loss = loss_function(logits, targets)

            epoch_loss_val += loss.item()

        epoch_loss_val /= len(loader_val)
        val_losses.append(epoch_loss_val)

        print(
            f"Epoch {len(train_losses)}: train loss {epoch_loss_train:.3f} | val loss {epoch_loss_val:.3f}"
        )

    return train_losses, val_losses
