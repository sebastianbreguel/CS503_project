import argparse
import ast
import pprint
import json
import torch
import yaml
from torchsummary import summary
import time
from dataset import get_dataset
from utils import (
    get_device,
    get_loss,
    get_model,
    get_optimizer,
    test_model,
    train_model,
)


def main(config):
    # print configuration
    pprint.pprint(config)

    # device
    device = get_device()
    print(device)

    # loss
    loss = get_loss(config["training"]["loss"])

    # model
    model = get_model(config)
    # load model
    model.load_state_dict(torch.load("weights/ViT/best_model_Wed_May_31_12_35_08_2023.pth"))
    # save time and hour

    # model.save_weights("weights/robust/" + localtime + ".pth")
    # load weights

    # Optimizer
    optimizer = get_optimizer(config, model.parameters())

    # input_size = ast.literal_eval(config["dataset"]["img_size"])
    # # # pri   nt(model)
    # summary(model, input_size)
    model = model.to(device)
    model_name = config["model"]["name"]

    # # # TODO: put this logic in an Algorithm class
    num_epochs = config["training"]["num_epochs"]
    loader_train, loader_val, loader_test = get_dataset(config["dataset"]["name"])
    # model, train_losses, val_losses, train_accuracy, val_accuracy = train_model(
    #     model,
    #     optimizer,
    #     loader_train,
    #     loader_val,
    #     num_epochs,
    #     loss,
    #     device,
    #     model_name=model_name,
    # )

    # localtime = time.asctime(time.localtime(time.time()))
    # localtime = localtime.replace(" ", "_")
    # localtime = localtime.replace(":", "_")
    # # test_loss, test_accuracy, test_corrupted_loss, test_corrupted_accuracy = test_model(model, loader_test, loss, device, model_name=model_name)
    test_loss, test_accuracy = test_model(model, loader_test, loss, device, model_name=model_name)
    # with open(f"weights/{model_name}/" + localtime + ".json", "w") as outfile:
    #     json.dump(
    #         {
    #             "train_losses": train_losses,
    #             "val_losses": val_losses,
    #             "train_accuracy": train_accuracy,
    #             "val_accuracy": val_accuracy,
    #             "test_loss": test_loss,
    #             "test_accuracy": test_accuracy,
    #             # "test_corrupted_loss": test_corrupted_loss,
    #             # "test_corrupted_accuracy": test_corrupted_accuracy,
    #         },
    #         outfile,
    #     )


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
