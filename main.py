"""Main entrypoint for running, loading, saving and evaluating models."""

import argparse
import json
import random
from pathlib import Path, PurePath

import jsons
import torch
from termcolor import colored

from graphgen.config import Config
from graphgen.datasets.gqa.images import GQAImages
from graphgen.datasets.gqa.objects import GQAObjects
from graphgen.datasets.gqa.questions import GQAQuestions
from graphgen.datasets.gqa.scene_graphs import GQASceneGraphs
from graphgen.datasets.gqa.spatial import GQASpatial
from graphgen.utilities.serialisation import path_deserializer, path_serializer
from graphgen.utilities.visualisation import plot_image, plot_spatial_features


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
    print(colored("Environment:", attrs=["bold"]))
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")  # type: ignore
    print(f"device: {torch.cuda.get_device_name(device) if cuda else 'CPU'}")

    print(config)
    questions = GQAQuestions(
        config.dataset.filemap, config.dataset.split, config.dataset.version,
    )
    graphs = GQASceneGraphs(config.dataset.filemap, config.dataset.split)
    spatial = GQASpatial(config.dataset.filemap)
    objects = GQAObjects(config.dataset.filemap)
    images = GQAImages(config.dataset.filemap)

    qn_idx = random.randint(0, len(questions) - 1)
    question = questions[qn_idx]
    print(f"{question=}")
    graph = graphs[graphs.key_to_index(question["imageId"])]
    print(f"{graph=}")
    spatial_features = spatial[spatial.key_to_index(question["imageId"])]
    print({key: val.shape for key, val in spatial_features.items()})
    object_features = objects[objects.key_to_index(question["imageId"])]
    print({key: val.shape for key, val in object_features.items()})
    image = images[images.key_to_index(question["imageId"])]
    print(image.shape)
    # Plot visual features
    plot_image(image, Path("image.png"), object_features["bboxes"])
    plot_spatial_features(spatial_features["features"], Path("spatial.png"))


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


def load_config(filename: str) -> Config:
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
    return config


if __name__ == "__main__":
    parsed_args = parse_args()
    config_obj = load_config(parsed_args.config)
    main(config_obj)
