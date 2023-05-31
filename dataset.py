import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageFolder, Food101, ImageNet
import numpy as np
import skimage as sk
from skimage.filters import gaussian
import cv2


BATCH_SIZE = 32

CORRUPTIONS = [
    "identity",
    "shot_noise",
    "impulse_noise",
    "gaussian_noise",
    "gaussian_blur",
    "glass_blur",
]


def add_corruption(x, corruption_type, severity=1):
    """
    Some of this corruption code is from:
    https://github.com/hendrycks/robustness/blob/master/ImageNet-C/imagenet_c/imagenet_c/corruptions.py
    but it has been adapted and generalized to a batch of images of any size.
    args:
        inputs: a batch of images. shape (batch_size, C, H, W)
        corruption_type: the type of corruption to add
        severity: the severity of the corruption. 1, 2, 3, 4, or 5.
    returns:
        the corrupted images
    """
    device = x.device
    x = x.cpu().numpy() / 255.0
    x = np.nan_to_num(x)
    if corruption_type == "identity":
        pass
    elif corruption_type == "shot_noise":
        c = [60, 25, 12, 5, 3][severity - 1]
        x = np.clip(np.random.poisson(x * c) / float(c), 0, 1)
    elif corruption_type == "impulse_noise":
        c = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
        x = sk.util.random_noise(x, mode="s&p", amount=c)
    elif corruption_type == "gaussian_noise":
        c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
        x = np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1)
    elif corruption_type == "gaussian_blur":
        c = [1, 2, 3, 4, 6][severity - 1]
        for i in range(x.shape[0]):
            x[i] = gaussian(x[i], sigma=c, channel_axis=0)
    elif corruption_type == "glass_blur":
        c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
        for i in range(x.shape[0]):
            x[i] = gaussian(x[i], sigma=c[0], channel_axis=0)
            h, w = x[i].shape[1:]
            for _ in range(c[2]):
                for hh in range(h - c[1], c[1], -1):
                    for ww in range(w - c[1], c[1], -1):
                        dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                        hh_prime, ww_prime = hh + dy, ww + dx
                        if hh_prime >= 0 and hh_prime < h and ww_prime >= 0 and ww_prime < w:
                            x[i, :, hh, ww], x[i, :, hh_prime, ww_prime] = x[i, :, hh_prime, ww_prime].copy(), x[i, :, hh, ww].copy()
            x[i] = gaussian(x[i], sigma=c[0], channel_axis=0)
    else:
        raise ValueError(f"Invalid corruption type: {corruption_type}")
    # Clip to ensure values are within the correct range and convert back to PyTorch tensor
    x = np.clip(x, 0, 1)
    # Convert to PyTorch tensor and scale back to original range (0-255) if needed.
    x = torch.from_numpy(x).float().to(device) * 255.0
    return x


def get_dataset(
    name: str,
    num_workers: int = 1,
):
    if name == "MNIST":
        transform = torchvision.transforms.Compose(
            [
                # TODO resize
                torchvision.transforms.ToTensor(),
            ]
        )

        dataset_train_val = MNIST(root="./data", train=True, download=True, transform=transform)
        dataset_train, dataset_val = torch.utils.data.random_split(dataset_train_val, [50_000, 10_000])
        dataset_test = MNIST(root="./data", train=False, download=True, transform=transform)

    elif name == "MNIST-C":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )
        folder = "./data/mnist-c/"
        dataset_train_val = ImageFolder(
            root=folder + "train",
            transform=transform,
        )

    elif name == "CIFAR10":
        transform = transforms.Compose(
            [
                # TODO resize
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        dataset_train_val = CIFAR10(root="./data", train=True, download=True, transform=transform)
        dataset_train, dataset_val = torch.utils.data.random_split(dataset_train_val, [45_000, 5_000])

        dataset_test = CIFAR10(root="./data", train=False, download=True, transform=transform)

    elif name == "CIFAR100":
        transform_train = transforms.Compose(
            [
                # TODO resize
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
            ]
        )
        transform_test = transforms.Compose(
            [
                torchvision.transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
            ]
        )

        dataset_train_val = CIFAR100(root="./data", train=True, download=True, transform=transform_train)
        dataset_test = CIFAR100(root="./data", train=False, download=True, transform=transform_test)

        dataset_train, dataset_val = torch.utils.data.random_split(dataset_train_val, [45_000, 5_000])
        # TODO Imagnet
        # TODO Cifar10-100  and Imagnet C
    elif name == "FOOD101":
        transform_train = transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                torchvision.transforms.AugMix(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        dataset_train_val = Food101(
            root="./data",
            split="train",
            download=True,
            transform=transform_train,
        )
        dataset_test = Food101(
            root="./data",
            split="test",
            download=True,
            transform=transform_test,
        )
        dataset_train, dataset_val = torch.utils.data.random_split(dataset_train_val, [75750 - 7575, 7575])

        dataset_train, dataset_val = torch.utils.data.random_split(dataset_train_val, [75750 - 7575, 7575])

    loader_train = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
    )
    loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, num_workers=num_workers, drop_last=False)
    loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, num_workers=num_workers, drop_last=False)

    return loader_train, loader_val, loader_test
