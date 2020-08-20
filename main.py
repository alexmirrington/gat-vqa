"""Main entrypoint for running, loading, saving and evaluating models."""

import argparse
import json
from pathlib import Path, PurePath
from typing import Any, Tuple

import jsons
import numpy as np
import stanza
import torch
import wandb
from termcolor import colored
from torch_geometric.data import DataLoader

from graphgen.config import Config
from graphgen.datasets.factory import DatasetFactory
from graphgen.datasets.utilities import ChunkedRandomSampler
from graphgen.modules.gcn import GCN
from graphgen.utilities.serialisation import path_deserializer, path_serializer


def main(config: Config) -> None:
    """Run a model according to the config parameters.

    Params:
    -------
    `config`: A configuration object defining configuration information such as
    model parameters and other important settings for the current run.

    Returns:
    --------
    None.
    """
    # pylint: disable=too-many-locals

    # Download and initialise resources
    print(colored("initialisation:", attrs=["bold"]))
    stanza.download(lang="en")

    # Print environment info
    print(colored("environment:", attrs=["bold"]))
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print(f"device: {torch.cuda.get_device_name(device) if cuda else 'CPU'}")
    print(config)

    # Preprocess data
    print(colored("preprocessing:", attrs=["bold"]))
    factory = DatasetFactory()
    dataset = factory.create(config)

    print(colored("model:", attrs=["bold"]))
    model = GCN((300, 600, 1200, 1878))  # 1878 is number of unique answers
    model.to(device)
    model.train()
    print(f"{model=}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    print(f"{optimizer=}")
    criterion = torch.nn.NLLLoss()
    print(f"{criterion=}")

    # Run model
    print(colored("running:", attrs=["bold"]))
    sampler = ChunkedRandomSampler(dataset.questions)
    dataloader = DataLoader(
        dataset,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.workers,
        sampler=sampler,
    )
    num_epochs = 100
    for epoch in range(num_epochs):
        correct = 0
        for batch, sample in enumerate(dataloader):
            question = sample["question"]
            data = question["dependencies"].to(device)
            targets = question["answer"].to(device)
            optimizer.zero_grad()
            preds = model(data)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            preds_np = np.argmax(preds.detach().cpu().numpy(), axis=1)
            targets_np = targets.detach().cpu().numpy()
            correct += np.sum(np.equal(preds_np, targets_np))

            # Calculate and log metrics
            if batch % config.logging.step == config.logging.step - 1:
                train_acc = correct / ((batch + 1) * dataloader.batch_size)
                print(
                    colored("batch:", attrs=["bold"]),
                    f"{batch + 1}/{len(dataloader)}",
                    colored("loss:", attrs=["bold"], color="cyan"),
                    f"{loss:.4f}",
                    colored("acc:", attrs=["bold"], color="green"),
                    f"{train_acc:.4f}",
                    end="\r",
                )
                wandb.log(
                    {
                        "epoch": epoch + batch / len(dataloader),
                        "train/loss": loss,
                        "train/accuracy": train_acc,
                    }
                )
        wandb.log({"epoch": epoch + 1, "train/loss": loss, "train/accuracy": train_acc})
        print(
            colored("epoch:", attrs=["bold"]),
            f"{epoch + 1}",
            colored("loss:", attrs=["bold"], color="cyan"),
            f"{loss:.4f}",
            colored("acc:", attrs=["bold"], color="green"),
            f"{train_acc:.4f}",
        )


def parse_args() -> argparse.Namespace:
    """Parse `sys.argv` and return an `argparse.Namespace` object.

    Params:
    -------
    None.

    Returns:
    --------
    An `argparse.Namespace` object containing the values of each parsed
    argument.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="The config to load settings from."
    )
    return parser.parse_args()


def load_config(filename: str) -> Tuple[Config, Any]:
    """Load a JSON configuration from `filename` into a `Config` object.

    Params:
    -------
    `filename`: The file to load the config from. Must be a valid JSON-formatted
    file.

    Returns:
    --------
    A `Config` object containing the deserialised JSON data.
    """
    with open(filename, "r") as file:
        config_json = json.load(file)

    # Set up custom `pathlib.Path` serialisers and deserialisers.
    jsons.set_deserializer(path_deserializer, PurePath)
    jsons.set_serializer(path_serializer, PurePath)

    config: Config = jsons.load(config_json, Config)
    return config, config_json


if __name__ == "__main__":
    parsed_args = parse_args()
    config_obj, config_dict = load_config(parsed_args.config)
    if not Path(".wandb").exists():
        Path(".wandb").mkdir()
    wandb.init(project="graphgen", dir=".wandb", config=config_dict)
    main(config_obj)
