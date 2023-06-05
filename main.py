import argparse
import ast
import pprint
import json
import torch
import numpy
import random
import yaml
from torchsummary import summary
import time
from dataset import get_dataset, get_dataset_to_corrupt
from utils import (
    get_device,
    get_loss,
    get_model,
    get_optimizer,
    test_model,
    train_model,
    test_corruptions,
)


def main(config):
    # print configuration
    pprint.pprint(config)
    random.seed(42)
    numpy.random.seed(42)
    torch.manual_seed(42)

    # device
    device = get_device()
    print(device)

    # loss
    loss = get_loss(config["training"]["loss"])

    # model
    model_name = config["model"]["name"]
    model = get_model(config, model_name, device)

    # Optimizer
    optimizer = get_optimizer(config, model.parameters())

    # input_size = ast.literal_eval(config["dataset"]["img_size"])
    # summary(model, input_size)
    model = model.to(device)

    loader_train, loader_val, loader_test = get_dataset(config["dataset"]["name"])

    # Training
    num_epochs = config["training"]["num_epochs"]
    model, train_losses, val_losses, train_accuracy, val_accuracy = train_model(
        model,
        optimizer,
        loader_train,
        loader_val,
        num_epochs,
        loss,
        device,
        model_name=model_name,
    )

    # Testing Normal
    test_loss, test_accuracy, test_accuracy_5 = test_model(model, loader_test, loss, device, model_name=model_name)

    # Testing Corruptions
    test_corruptions(model, loss, device, model_name=model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust ViT")
    parser.add_argument(
        "--config",
        default="yamls/naive.yaml",
        type=str,
        help="Path to a .yaml config file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)
