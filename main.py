"""Main entrypoint for running, loading, saving and evaluating models."""

import argparse
import json
import os
from enum import Enum
from pathlib import Path, PurePath
from typing import Iterable, List, Optional, Tuple

import jsons
import stanza
import torch
from termcolor import colored

import wandb
from graphgen.config import Config
from graphgen.utilities.factories import (
    DatasetFactory,
    PreprocessingFactory,
    RunnerFactory,
)
from graphgen.utilities.runners import ResumeInfo
from graphgen.utilities.serialisation import path_deserializer, path_serializer


class JobType(Enum):
    """Enum specifying possible job types for the current run."""

    PREPROCESS = "preprocess"
    TRAIN = "train"
    TEST = "test"


def main(args: argparse.Namespace, config: Config) -> None:
    """Run a model according to the config parameters.

    Params:
    -------
    `config`: A configuration object defining configuration information such as
    model parameters and other important settings for the current run.

    Returns:
    --------
    None.
    """
    # Notes:
    # - 1878 is the number of unique answers from the GQA paper
    # - 1843 is the number of answers across train, val and testdev

    # Download and initialise resources
    print(colored("initialisation:", attrs=["bold"]))
    stanza.download(lang="en", dir=".stanza")

    # Print environment info
    print(colored("environment:", attrs=["bold"]))
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print(f"device: {torch.cuda.get_device_name(device) if cuda else 'CPU'}")
    print(config)

    if args.job == JobType.PREPROCESS:
        preprocess(config)
    elif args.job == JobType.TRAIN:
        resume = None
        if args.resume != "":
            run_id, checkpoint = args.resume.split(":")
            resume = ResumeInfo(run_id, checkpoint)
        run(config, device, resume)
    elif args.job == JobType.TEST:
        raise NotImplementedError()
    else:
        raise NotImplementedError()


def preprocess(config: Config) -> None:
    """Preprocess `config.dataset` according to the `config.preprocessing` config."""
    print(colored("preprocessing:", attrs=["bold"]))
    factory = PreprocessingFactory()
    factory.process(config)


def run(config: Config, device: torch.device, resume: Optional[ResumeInfo]) -> None:
    """Train a model according to the `config.model` config."""
    # Load datasets
    print(colored("loading datasets:", attrs=["bold"]))
    dataset_factory = DatasetFactory()
    datasets, preprocessors = dataset_factory.create(config)
    print(f"train: {len(datasets.train)}")
    print(f"val: {len(datasets.val)}")

    # Create model runner
    print(colored("model:", attrs=["bold"]))
    runner_factory = RunnerFactory()
    runner = runner_factory.create(config, device, preprocessors, datasets, resume)
    print(f"{runner.model=}")
    print(f"{runner.criterion=}")
    print(f"{runner.optimiser=}")

    # Run model
    print(colored("running:", attrs=["bold"]))
    runner.train()


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
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
    parser.add_argument(
        "--job",
        type=JobType,
        choices=list(iter(JobType)),
        metavar=str({str(job.value) for job in iter(JobType)}),
        help="Job type for this run.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="A wandb run and filename to resume training a model from, \
        e.g. graphgen/a1b2c3d:checkpoints/current.pt",
    )
    parser.add_argument(
        "--sync", action="store_true", help="Sync results to wandb if specified."
    )
    return parser.parse_known_intermixed_args()


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


def merge_config(args: Iterable[str], config: Config) -> Config:
    """Merge any leftover args of form param/nested_param=value into config object."""
    # Apply override args
    for arg in args:
        arg = arg.lstrip("-")
        param, value = arg.split("=")
        param_keys = param.split("/")
        subconfig = config
        for idx, key in enumerate(param_keys):
            try:
                if idx == len(param_keys) - 1:
                    # Cast value to type of field value and set attribute
                    field_type = type(getattr(subconfig, key))
                    try:
                        deserialised_value = jsons.loads(value, field_type)
                    except jsons.exceptions.DecodeError:
                        deserialised_value = field_type(value)
                    setattr(subconfig, key, deserialised_value)
                else:
                    # Get subconfig from key
                    subconfig = getattr(subconfig, key)
            except AttributeError as ex:  # Ensure the attribute exists
                raise ValueError(
                    f"Invalid argument {arg}. Could not merge with config object."
                ) from ex
            except (ValueError, TypeError) as ex:
                raise ValueError(
                    f"Invalid argument {arg}. Value could not be converted to \
                    type {field_type}"
                ) from ex
    return config


if __name__ == "__main__":
    # Parse config
    parsed_args, remaining_args = parse_args()
    config_obj = load_config(parsed_args.config)
    config_obj = merge_config(remaining_args, config_obj)
    # Set up wandb
    # os.environ["WANDB_SILENT"] = "false" if parsed_args.sync else "true"
    os.environ["WANDB_MODE"] = "run" if parsed_args.sync else "dryrun"
    if not Path(".wandb").exists():
        Path(".wandb").mkdir()
    wandb.init(
        project="graphgen",
        dir=".wandb",
        job_type=parsed_args.job.value,
        config=jsons.dump(config_obj),
    )
    # Run main with parsed config
    main(parsed_args, config_obj)
