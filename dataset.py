import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import (
    CIFAR10,
    CIFAR100,
    MNIST,
    ImageFolder,
    Food101,
    StanfordCars,
)


# class StanfordCars(torch.utils.data.Dataset):
#     def __init__(self, root_path, transform=None):
#         self.images = [os.path.join(root_path, file) for file in os.listdir(root_path)]
#         self.transform = transform

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, index):
#         image_file = self.images[index]
#         image = Image.open(image_file).convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#         return image[None]


BATCH_SIZE = 16

CORRUPTIONS = [
    "identity",
    "shot_noise",
    "impulse_noise",
    "glass_blur",
    "motion_blur",
    "shear",
    "scale",
    "rotate",
    "brightness",
    "translate",
    "stripe",
    "fog",
    "spatter",
    "dotted_line",
    "zigzag",
    "canny_edges",
]


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

        dataset_train_val = MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        dataset_train, dataset_val = torch.utils.data.random_split(
            dataset_train_val, [50_000, 10_000]
        )
        dataset_test = MNIST(
            root="./data", train=False, download=True, transform=transform
        )

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
                # TODO resize
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                torchvision.transforms.Resize((32, 32)),
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
        # TODO Imagnet
        # TODO Cifar10-100  and Imagnet C
    elif name == "FOOD101":
        transform_train = transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                torchvision.transforms.AugMix(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
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
        dataset_train, dataset_val = torch.utils.data.random_split(
            dataset_train_val, [75750 - 7575, 7575]
        )

        dataset_train, dataset_val = torch.utils.data.random_split(
            dataset_train_val, [75750 - 7575, 7575]
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
