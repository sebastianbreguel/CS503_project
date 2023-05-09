from utils import (
    get_device,
    get_model,
    get_loss,
    get_optimizer,
    train_model,
    test_model,
)
from dataset import get_dataset


import torch
from torchsummary import summary
import argparse
import yaml
import pprint
import ast


def main(config):
    #print configuration
    pprint.pprint(config)

    # device
    device = get_device()
    print(device)
    
    # loss
    loss = get_loss(config["training"]["loss"])

    # model
    model = get_model(config)
    
    #Optimizer
    optimizer = get_optimizer(config, model.parameters())

    input_size = ast.literal_eval(config["dataset"]["img_size"])
    summary(model, input_size)
    model = model.to(device)

    #TODO: put this logic in an Algorithm class
    num_epochs = config["training"]["num_epochs"]
    loader_train, loader_val, loader_test = get_dataset(config["dataset"]["name"])
    model, _, _ = train_model(
        model, optimizer, loader_train, loader_val, num_epochs, loss, device
    )
    
    test_model(model, loader_test, loss, device)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    # parser.add_argument("--dataset", type=str, help="", default="MNIST")
    # parser.add_argument("--epoch", type=int, help="", default=2)
    # parser.add_argument("--loss", action="store", help="Loss to use", default="CE")
    # parser.add_argument("--model", action="store", help="Model to use")
    # parser.add_argument("--optimizer", action="store", help="Optimizer to use")

    # results = parser.parse_args()

    # dataset = results.dataset
    # num_epochs = results.epoch
    # loss = results.loss
    # model = results.model
    # optimizer = results.optimizer

    parser = argparse.ArgumentParser(description="Robust ViT")
    parser.add_argument(
        "--config", default="yamls/naive.yaml", type=str, help="Path to a .yaml config file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)
