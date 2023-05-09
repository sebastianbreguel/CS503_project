import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, ImageNet
import torch.nn.functional as F
import torchvision.transforms as transforms
from models import ViT
from params import BATCH_SIZE, IMG_SIZE, LR
from tqdm import tqdm


def get_model(name: str = "ViT"):
    if name == "ViT":
        model = ViT(
            # img_size=IMG_SIZE,
            img_size=14,
            patch_size=2,
            in_channels=3,
            embed_dim=192,
            num_classes=100,
            depth=2,
            num_heads=2,
            mlp_ratio=4.0,
            head_bias=False,
            drop=0.1,
            # patch_size=2,
            # in_channels=3,
            # embed_dim=192,
            # num_classes=10,
            # depth=5,
            # num_heads=4,
            # mlp_ratio=4.0,
            # drop=0.15,
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
    model,
    optimizer,
    loader_train,
    loader_val,
    num_epochs,
    loss_function,
    device: str = "cpu",
):
    train_losses = []
    val_losses = []

    for _ in range(num_epochs):
        # Train loop
        model.train()
        epoch_loss_train = 0
        for imgs, cls_idxs in tqdm(loader_train, total=len(loader_train)):
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

    return model, train_losses, val_losses


def test_model(model, loader_test, loss_function, device: str = "cpu"):
    test_loss = 0
    correct = 0

    model.eval()
    for imgs, cls_idxs in loader_test:
        inputs, targets = imgs.to(device), cls_idxs.to(device)

        with torch.no_grad():
            logits = model(inputs)
        loss = loss_function(logits, targets)
        test_loss += loss.item()

        pred = logits.argmax(dim=1, keepdim=True)
        correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(loader_test)
    accuracy = correct / len(loader_test.dataset)

    print(f"Test loss: {test_loss:.3f}")
    print(f"Test top-1 accuracy: {accuracy*100}%")
