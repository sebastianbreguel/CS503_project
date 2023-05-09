import torch
from utils import (
    get_device,
    get_model,
    get_loss,
    get_optimizer,
    train_model,
    test_model,
)
from dataset import get_dataset
import argparse

from params import IMG_SIZE


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, help="", default="MNIST")
    parser.add_argument("--epoch", type=int, help="", default=2)
    parser.add_argument("--loss", action="store", help="Loss to use", default="CE")
    parser.add_argument("--model", action="store", help="Model to use")
    parser.add_argument("--optimizer", action="store", help="Optimizer to use")

    results = parser.parse_args()

    dataset = results.dataset
    num_epochs = results.epoch
    loss = results.loss
    model = results.model
    optimizer = results.optimizer

    device = get_device()
    loader_train, loader_val, loader_test = get_dataset(dataset)
    loss, model = get_loss(loss), get_model(model).to(device)
    optimizer = get_optimizer(optimizer, model.parameters())

    model, train_score, eval_score = train_model(
        model, optimizer, loader_train, loader_val, num_epochs, loss, device
    )

    test_model(model, loader_test, loss, device)
