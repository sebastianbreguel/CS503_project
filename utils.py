import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, ImageNet
import torch.nn.functional as F
import torchvision.transforms as transforms
from models import ViT
from params import  BATCH_SIZE, IMG_SIZE, LR


def get_model(name: str = "ViT"):
    if name == "ViT":
        model = ViT(
            img_size=IMG_SIZE,
            patch_size=2,
            in_channels=3,
            embed_dim=192,
            num_classes=10,
            depth=5,
            num_heads=4,
            mlp_ratio=4.0,
            drop=0.15,
        )

    num_parameters = sum([p.numel() for p in model.parameters()])
    print(f"Number of parameters: {num_parameters:,}")

    return model


# parameters will be a generator
def get_optimizer(name, parameters, lr: float = LR):

    if name == "Adam":
        optimizer = torch.optim.Adam(parameters, lr=lr)

    elif name == "AdamW":
        optimizer = torch.optim.AdamW(parameters, lr=lr)

    elif name == "SGD":
        optimizer = torch.optim.SGD(parameters, lr=lr)

    return optimizer


def get_dataset(
    name,
    num_workers: int = 8,
):
    if name == "MNIST":

        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
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
    elif name == "CIFAR10":
        transform = transforms.Compose(
            [
                torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        dataset_train_val = CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        dataset_train, dataset_val = torch.utils.data.random_split(
            dataset_train_val, [45_000, 5_000]
        )

        dataset_test = CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )

    elif name == "CIFAR100":
        transform_train = transforms.Compose([
            torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomCrop(IMG_SIZE, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])
        transform_test = transforms.Compose([
            torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])

        dataset_train_val = CIFAR100(
            root="./data", train=True, download=True, transform=transform_train
        )
        dataset_train, dataset_val = torch.utils.data.random_split(
            dataset_train_val, [45_000, 5_000]
        )

        dataset_test = CIFAR100(
            root="./data", train=False, download=True, transform=transform_test
        )



    loader_train = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
    )
    loader_val = DataLoader(
        dataset_val, 
        batch_size=BATCH_SIZE, 
        num_workers=num_workers, 
        drop_last=False
    )
    loader_test = DataLoader(
        dataset_test, 
        batch_size=BATCH_SIZE, 
        num_workers=num_workers, 
        drop_last=False
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


def get_loss(name: str = "CE"):
    if name == "CE":
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

            optimizer.zero_grad()
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
