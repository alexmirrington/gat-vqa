"""Main entrypoint for running, loading, saving and evaluating models."""

import argparse
import json
import random
from pathlib import PurePath
from typing import Any, Tuple

import jsons
import torch
from termcolor import colored

import wandb
from graphgen.config import Config
from graphgen.datasets.gqa import GQA
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
    # Print environment info
    print(colored("environment:", attrs=["bold"]))
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")  # type: ignore
    print(f"device: {torch.cuda.get_device_name(device) if cuda else 'CPU'}")

    print(config)
    dataset = GQA(config.dataset.filemap, config.dataset.split, config.dataset.version)

    idx = random.randint(0, len(dataset) - 1)
    question, image, spatial, objects, graph = dataset[idx]
    print(f"{question=}")
    print(f"{graph=}")
    print({key: val.shape for key, val in spatial.items()})
    print({key: val.shape for key, val in objects.items()})
    print(image.shape)


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
    wandb.init(project="graphgen", config=config_dict)
    main(config_obj)
