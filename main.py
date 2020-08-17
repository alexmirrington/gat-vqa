"""Main entrypoint for running, loading, saving and evaluating models."""

import argparse
import json
from pathlib import Path, PurePath
from typing import Any, Tuple

import jsons
import torch
import wandb
from termcolor import colored
from torch.utils.data import DataLoader
from tqdm import tqdm

from graphgen.config import Config
from graphgen.datasets.gqa import GQA, GQAImages, GQAQuestions
from graphgen.datasets.utilities import ChunkedRandomSampler
from graphgen.utilities.preprocessing import QuestionPreprocessor
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
    device = torch.device("cuda" if cuda else "cpu")
    print(f"device: {torch.cuda.get_device_name(device) if cuda else 'CPU'}")
    print(config)

    # Preprocess data
    print(colored("preprocessing:", attrs=["bold"]))
    questions = GQAQuestions(
        config.dataset.filemap,
        config.dataset.split,
        config.dataset.version,
        preprocessor=QuestionPreprocessor(),
        tempdir=Path(),
    )
    print(f"loaded {questions.__class__.__name__}")

    images = GQAImages(config.dataset.filemap)
    print(f"loaded {images.__class__.__name__}")

    dataset = GQA(questions, images=images)
    print(f"loaded {dataset.__class__.__name__}")

    # Run model
    print(colored("running:", attrs=["bold"]))
    sampler = ChunkedRandomSampler(dataset.questions)
    dataloader = DataLoader(
        dataset,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.workers,
        sampler=sampler,
        collate_fn=lambda batch: list(zip(*batch)),
    )
    for sample in tqdm(dataloader, desc="batch: "):
        question, image, spatial, objects, boxes, scene_graph = sample


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
