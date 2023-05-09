from models import ViT
from utils import get_dataset, get_device, get_loss, train_model

import torch
from torchsummary import summary
import argparse
import yaml
import pprint
import ast


MODELS = ["ViT"]
OPTIMIZERS = ["AdamW"]

def main(config):
    #print configuration
    pprint.pprint(config)

    # device
    device = get_device()

    # model
    if config["model"]["name"] == "ViT":
        model = ViT(**config["model"]).to(device)
    else:
        return NotImplementedError("Model not implemented. Please choose from: " + str(MODELS))
    input_size = ast.literal_eval(config["dataset"]["img_size"])
    summary(model, input_size)

    # optimizer
    if config["optimizer"]["name"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"]["params"])
    else:
        return NotImplementedError("Optimizer not implemented. Please choose from: " + str(OPTIMIZERS))

    #TODO: put this logic in an Algorithm class
    num_epochs = config["training"]["num_epochs"]
    loss = get_loss(config["training"]["loss"])
    loader_train, loader_val, loader_test = get_dataset(config["dataset"]["name"])
    train_model(model, optimizer, loader_train, loader_val, num_epochs, loss, device)
    #TODO: add test function


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Robust ViT")
    parser.add_argument(
        "--config", default="yamls/naive.yaml", type=str, help="Path to a .yaml config file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)