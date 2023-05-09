import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, ImageNet
import torchvision.transforms as transforms
from params import BATCH_SIZE, IMG_SIZE


def get_dataset(
    name: str,
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

        transform_train = transforms.Compose(
            [
                torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomCrop(IMG_SIZE, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
                ),
            ]
        )

        dataset_train_val = CIFAR100(
            root="./data", train=True, download=True, transform=transform_train
        )
        dataset_test = CIFAR100(
            root="./data", train=False, download=True, transform=transform_test
        )

        dataset_train, dataset_val = torch.utils.data.random_split(
            dataset_train_val, [45_000, 5_000]
        )
    elif name == "ImageNet":
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        dataset_train = ImageNet(
            root="./data", split="train", download=True, transform=transform
        )
        dataset_val = ImageNet(
            root="./data", split="val", download=True, transform=transform
        )
        dataset_test = ImageNet(
            root="./data", split="test", download=True, transform=transform
        )

    loader_train = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
    )
    loader_val = DataLoader(
        dataset_val, batch_size=BATCH_SIZE, num_workers=num_workers, drop_last=False
    )
    loader_test = DataLoader(
        dataset_test, batch_size=BATCH_SIZE, num_workers=num_workers, drop_last=False
    )

    return loader_train, loader_val, loader_test
