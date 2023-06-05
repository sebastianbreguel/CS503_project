import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
from models import MedViT, FoodViT
from robust import RVT
from transformers import ViTConfig, ViTForImageClassification
import random
from torchvision import transforms
import uuid
import os
import torchvision.transforms as transformspil
from dataset import get_dataset_to_corrupt

from dataset import add_corruption

MODELS = ["ViT", "RVT", "MedViT", "FoodViT"]
OPTIMIZERS = ["AdamW", "Adam", "SGD"]


def get_model(config, model_name, device) -> torch.nn.Module:
    if model_name == "ViT":
        configuration = ViTConfig(**config["model"]["params"])
        model = ViTForImageClassification(configuration)
        model.load_state_dict(
            torch.load(
                "weights/ViT/best_model_24_Sun_Jun__4_21_45_29_2023.pth",
                map_location=device,
            )
        )

    elif model_name == "MedViT":
        model = MedViT(**config["model"]["params"])

    elif model_name == "FoodViT":
        model = FoodViT(**config["model"]["params"])
        model.load_state_dict(
            torch.load(
                "weights/FoodViT/best_model.pth",
                map_location=device,
            )
        )

    elif model_name == "RVT":
        model = RVT(**config["model"]["params"])
        model.load_state_dict(
            torch.load(
                "weights/RVT/best_model.pth",
                map_location=device,
            )
        )

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

    for _ in range(8, 8 + num_epochs):
        # Train loop
        model.train()
        epoch_loss_train = 0
        correct = 0
        correct_5 = 0
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
            correct_5 += torch.topk(logits, 5, dim=1)[1].eq(targets.view(-1, 1)).sum().item()

        train_accuracy = correct / len(loader_train.dataset)
        epoch_loss_train /= len(loader_train)
        train_accuracys.append(train_accuracy)
        train_losses.append(epoch_loss_train)
        correct = 0
        correct_5_val = 0

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
            correct_5_val += torch.topk(logits, 5, dim=1)[1].eq(targets.view(-1, 1)).sum().item()

        val_accuracy = correct / len(loader_val.dataset)

        epoch_loss_val /= len(loader_val)
        val_accuracys.append(val_accuracy)
        val_losses.append(epoch_loss_val)
        if val_accuracy > best_acc_train:
            best_acc_train = val_accuracy
            best_model = model.state_dict()
            # save weights
            torch.save(best_model, f"weights/{model_name}/best_model_{_}_{localtime}.pth")

        print(f"Epoch {_}: train loss {epoch_loss_train:.3f} | val loss {epoch_loss_val:.3f}")
        print(f"Epoch {_}: train accuracy {train_accuracy*100:.3f}% | val accuracy {val_accuracy*100:.3f}%")
        print(f"Epoch {_}: train top 5 accuracy {correct_5/len(loader_train.dataset)*100:.3f}% | val top 5 accuracy {correct_5_val/len(loader_val.dataset)*100:.3f}%")

    return model, train_losses, val_losses, train_accuracys, val_accuracys


def test_model(model, loader_test, loss_function, device: str = "cpu", model_name: str = "ViT"):
    test_loss = 0
    correct = 0
    correct_5 = 0

    model.eval()

    for imgs, cls_idxs in tqdm(loader_test, total=len(loader_test)):
        inputs, targets = imgs.to(device), cls_idxs.to(device)

        # Add corruptions to inputs

        with torch.no_grad():
            logits = model(inputs)
            if model_name == "ViT":
                logits = logits.logits
        loss = loss_function(logits, targets)
        test_loss += loss.item()

        pred = logits.argmax(dim=1, keepdim=True)
        correct += pred.eq(targets.view_as(pred)).sum().item()
        correct_5 += torch.topk(logits, 5, dim=1)[1].eq(targets.view(-1, 1)).sum().item()

    test_loss /= len(loader_test)
    accuracy = correct / len(loader_test.dataset)
    accuracy_5 = correct_5 / len(loader_test.dataset)

    print(f"Test loss: {test_loss:.3f}")
    print(f"Test top-1 accuracy: {accuracy*100}%")
    print(f"Test top-5 accuracy: {accuracy_5*100}%")

    return (
        test_loss,
        accuracy,
        accuracy_5,
    )


def store_corruptions(loader_test):
    # Transform tensor to PIL Image
    to_pil = transformspil.ToPILImage()
    print("Storing corruptions...")
    random.seed(42)
    for inputs, targets in tqdm(loader_test, total=len(loader_test)):
        # Add corruptions to inputs
        corruption_type = random.choice(
            [
                "shot_noise",
                "impulse_noise",
                "gaussian_noise",
                "gaussian_blur",
                "glass_blur",
            ]
        )
        severity = random.randint(1, 5)
        inputs = add_corruption(inputs, corruption_type, severity)

        # We are assuming inputs and targets are batches and they are tensors
        for img, target in zip(inputs, targets):
            # Create directory for each class if it doesn't exist
            class_dir = os.path.join("data", "food-101", "corrupted", str(target.item()))
            os.makedirs(class_dir, exist_ok=True)

            # Convert tensor to PIL Image
            img_pil = to_pil(img)

            # Save image to appropriate directory with a unique name
            img_name = f"{uuid.uuid4().hex}.png"  # Use UUID to ensure the uniqueness of the image names
            img_pil.save(os.path.join(class_dir, img_name))


def test_corruptions(model, loss_function, device: str = "cpu", model_name: str = "ViT"):
    test_corrupted_loss = 0
    correct_corrupted = 0

    model.eval()
    loader_test = get_dataset_to_corrupt()

    # Fix the seed for reproducibility
    random.seed(42)

    test_corrupted_loss = 0
    correct_corrupted = 0
    correct_5_corrupted = 0
    # save a list of corruption, serverity
    corruptions = []
    number = 0
    for imgs, cls_idxs in loader_test:
        inputs, targets = imgs.to(device), cls_idxs.to(device)
        corruption_type = random.choice(["shot_noise", "impulse_noise", "gaussian_noise", "gaussian_blur", "glass_blur"])

        # Add corruptions to inputs
        severity = random.randint(1, 5)
        now = time.time()
        corruptions.append([corruption_type, severity, time.time() - now])
        inputs = add_corruption(inputs, corruption_type, severity)
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        inputs = normalize(inputs)
        with torch.no_grad():
            logits = model(inputs)
            if model_name == "ViT":
                logits = logits.logits
        loss = loss_function(logits, targets)
        test_corrupted_loss += loss.item()

        pred = logits.argmax(dim=1, keepdim=True)
        correct_corrupted += pred.eq(targets.view_as(pred)).sum().item()
        correct_5_corrupted += torch.topk(logits, 5, dim=1)[1].eq(targets.view(-1, 1)).sum().item()

        if number % 100 == 0:
            print(number)
        number += 1

    # want to have the metrics
    # write over a txt file
    with open("corruptions.txt", "w") as f:
        for corruption in corruptions:
            f.write(f"{corruption[0]} {corruption[1]}\n")

    test_corrupted_loss /= len(loader_test)
    accuracy_corrupted = correct_corrupted / len(loader_test.dataset)
    accuracy_5_corrupted = correct_5_corrupted / len(loader_test.dataset)
    print(f"Test corrupted loss: {test_corrupted_loss:.3f}")
    print(f"Test corrupted top-1 accuracy: {accuracy_corrupted*100}%\n")
    print(f"Test corrupted top-5 accuracy: {accuracy_5_corrupted*100}%\n")
    print("-" * 50)

    return test_corrupted_loss, accuracy_corrupted, accuracy_5_corrupted
