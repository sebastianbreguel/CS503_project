import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageNet
from tqdm import tqdm

from models import BreguiT, ViT

MODELS = ["ViT", "BreguiT"]
OPTIMIZERS = ["AdamW", "Adam", "SGD"]


def get_model(config):

    if config["model"]["name"] == "ViT":
        model = ViT(**config["model"]["params"])

    elif config["model"]["name"] == "BreguiT":
        model = BreguiT(**config["model"]["params"])

    else:
        return NotImplementedError(
            "Model not implemented. Please choose from: " + str(MODELS)
        )

    num_parameters = sum([p.numel() for p in model.parameters()])
    print(f"Number of parameters: {num_parameters:,}")

    return model


# parameters will be a generator
def get_optimizer(config, parameters):

    # optimizer
    if config["optimizer"]["name"] == "AdamW":
        optimizer = torch.optim.AdamW(parameters, **config["optimizer"]["params"])

    elif config["optimizer"]["name"] == "Adam":
        optimizer = torch.optim.Adam(parameters, **config["optimizer"]["params"])

    elif config["optimizer"]["name"] == "SGD":
        optimizer = torch.optim.SGD(parameters, **config["optimizer"]["params"])

    else:
        return NotImplementedError(
            "Optimizer not implemented. Please choose from: " + str(OPTIMIZERS)
        )

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


def get_loss(name):

    LOSSES = ["cross entropy"]
    if name == "cross entropy":
        loss = F.cross_entropy
    else:
        raise NotImplementedError(
            "Loss not implemented. Please choose from: " + str(LOSSES)
        )

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
