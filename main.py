"""Main entrypoint for running, loading, saving and evaluating models."""

import argparse
import json
import re
import sys
from pathlib import Path, PurePath
from typing import Any, Dict, Tuple

import jsons
import torch
import wandb
from termcolor import colored
from torch.utils.data import DataLoader
from tqdm import tqdm

from graphgen.config import Config
from graphgen.datasets.gqa import GQA, GQAImages, GQAQuestions
from graphgen.datasets.utilities import ChunkedRandomSampler
from graphgen.utilities.serialisation import path_deserializer, path_serializer


class QuestionPreprocessor:
    """Class for preprocessing questions."""

    KEY_MASK = ("imageId", "question", "answer")
    VOCAB_MASK = ("question", "answer")

    def __init__(self) -> None:
        """Create a `QuestionPreprocessor` instance."""
        self.word_to_index: Dict[str, int] = {}

    def __call__(self, question: Any) -> Any:
        """Preprocess a question sample."""
        # Filter out unused fields
        result = {key: val for key, val in question.items() if key in self.KEY_MASK}

        # Populate word_to_index dict
        for key, val in result.items():
            if key in self.VOCAB_MASK:
                lval = val.lower()
                lval = re.sub(r"[^\w\s]", "", lval)
                tokens = []
                for word in lval.split():
                    if word not in self.word_to_index.keys():
                        self.word_to_index[word] = len(self.word_to_index)
                    tokens.append(self.word_to_index[word])
                result[key] = tokens

        return result


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
    for batch, sample in enumerate(tqdm(dataloader, file=sys.stdout)):
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
