import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
from models import MedViT, Testion
from robust import PoolingTransformer
from transformers import ViTConfig, ViTForImageClassification
import random

from dataset import add_corruption

MODELS = ["ViT", "BreguiT", "RVT", "MedViT"]
OPTIMIZERS = ["AdamW", "Adam", "SGD"]


def get_model(config) -> torch.nn.Module:
    if config["model"]["name"] == "ViT":
        configuration = ViTConfig(**config["model"]["params"])
        model = ViTForImageClassification(configuration)

    elif config["model"]["name"] == "MedViT":
        model = MedViT(**config["model"]["params"])

    elif config["model"]["name"] == "testion":
        model = Testion(**config["model"]["params"])

    elif config["model"]["name"] == "robust":
        model = PoolingTransformer(**config["model"]["params"])

    else:
        return NotImplementedError("Model not implemented. Please choose from: " + str(MODELS))

    num_parameters = sum([p.numel() for p in model.parameters()])
    print(f"Number of parameters: {num_parameters:,}")

    return model


# parameters will be a generator
def get_optimizer(config, parameters) -> torch.optim.Optimizer:
    # optimizer
    if config["optimizer"]["name"] == "AdamW":
        optimizer = torch.optim.AdamW(parameters, **config["optimizer"]["params"])

    elif config["optimizer"]["name"] == "Adam":
        optimizer = torch.optim.Adam(parameters, **config["optimizer"]["params"])

    elif config["optimizer"]["name"] == "SGD":
        optimizer = torch.optim.SGD(parameters, **config["optimizer"]["params"])

    else:
        return NotImplementedError("Optimizer not implemented. Please choose from: " + str(OPTIMIZERS))

    return optimizer


def get_device() -> torch.device:
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
        raise NotImplementedError("Loss not implemented. Please choose from: " + str(LOSSES))

    return loss


def train_model(
    model,
    optimizer,
    loader_train,
    loader_val,
    num_epochs,
    loss_function,
    device: str = "cpu",
    model_name: str = "ViT",
):
    train_losses = []
    train_accuracys = []
    val_losses = []
    val_accuracys = []

    localtime = time.asctime(time.localtime(time.time()))
    localtime = localtime.replace(" ", "_")
    localtime = localtime.replace(":", "_")

    best_acc_train = -1

    for _ in range(num_epochs):
        # Train loop
        model.train()
        epoch_loss_train = 0
        correct = 0
        for imgs, cls_idxs in tqdm(loader_train, total=len(loader_train)):
            inputs, targets = imgs.to(device), cls_idxs.to(device)
            logits = model(inputs)
            if model_name == "ViT":
                logits = logits.logits
            loss = loss_function(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_train += loss.item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()

        train_accuracy = correct / len(loader_train.dataset)
        epoch_loss_train /= len(loader_train)
        train_accuracys.append(train_accuracy)
        train_losses.append(epoch_loss_train)
        correct = 0

        # Validation loop
        model.eval()
        epoch_loss_val = 0
        for imgs, cls_idxs in tqdm(loader_val, total=len(loader_val)):
            inputs, targets = imgs.to(device), cls_idxs.to(device)
            logits = model(inputs)
            if model_name == "ViT":
                logits = logits.logits
            loss = loss_function(logits, targets)

            epoch_loss_val += loss.item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()

        val_accuracy = correct / len(loader_val.dataset)

        epoch_loss_val /= len(loader_val)
        val_accuracys.append(val_accuracy)
        val_losses.append(epoch_loss_val)
        if val_accuracy > best_acc_train:
            best_acc_train = val_accuracy
            best_model = model.state_dict()
            # save weights
            torch.save(best_model, f"weights/{model_name}/best_model_{localtime}.pth")

        print(f"Epoch {len(train_losses)}: train loss {epoch_loss_train:.3f} | val loss {epoch_loss_val:.3f}")
        print(f"Epoch {len(train_losses)}: train accuracy {train_accuracy*100:.3f}% | val accuracy {val_accuracy*100:.3f}%")

    return model, train_losses, val_losses, train_accuracy, val_accuracy


def test_model(model, loader_test, loss_function, device: str = "cpu", model_name: str = "ViT"):
    test_loss = 0
    correct = 0

    test_corrupted_loss = 0
    correct_corrupted = 0

    model.eval()

    # Fix the seed for reproducibility
    random.seed(42)

    # Wrap loader with tqdm to create a progress bar
    for imgs, cls_idxs in tqdm(loader_test, total=len(loader_test)):
        inputs, targets = imgs.to(device), cls_idxs.to(device)

        with torch.no_grad():
            logits = model(inputs)
            if model_name == "ViT":
                logits = logits.logits
        loss = loss_function(logits, targets)
        test_loss += loss.item()

        pred = logits.argmax(dim=1, keepdim=True)
        correct += pred.eq(targets.view_as(pred)).sum().item()

    for imgs, cls_idxs in tqdm(loader_test, total=len(loader_test)):
        inputs, targets = imgs.to(device), cls_idxs.to(device)

        # Add corruptions to inputs
        corruption_type = random.choice(["identity", "shot_noise", "impulse_noise", "gaussian_noise", "gaussian_blur", "glass_blur"])
        severity = random.randint(1, 5)
        inputs = add_corruption(inputs, corruption_type, severity)

        with torch.no_grad():
            logits = model(inputs)
            if model_name == "ViT":
                logits = logits.logits
        loss = loss_function(logits, targets)
        test_corrupted_loss += loss.item()

        pred = logits.argmax(dim=1, keepdim=True)
        correct_corrupted += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(loader_test)
    accuracy = correct / len(loader_test.dataset)

    test_corrupted_loss /= len(loader_test)
    accuracy_corrupted = correct_corrupted / len(loader_test.dataset)
    print(f"Test loss: {test_loss:.3f}")
    print(f"Test top-1 accuracy: {accuracy*100}%")

    print(f"Test corrupted loss: {test_corrupted_loss:.3f}")
    print(f"Test corrupted top-1 accuracy: {accuracy_corrupted*100}%")
    return test_loss, accuracy, test_corrupted_loss, accuracy_corrupted
