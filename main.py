"""Main entrypoint for running, loading, saving and evaluating models."""

import argparse
import json
import math
import os
from enum import Enum
from pathlib import Path, PurePath
from typing import Any, Callable, Dict, Tuple

import jsons
import numpy as np
import stanza
import torch
import wandb
from termcolor import colored
from torch.utils.data import DataLoader

from graphgen.config import Config
from graphgen.datasets.collators import VariableSizeTensorCollator
from graphgen.datasets.utilities import ChunkedRandomSampler
from graphgen.metrics import Metric, MetricCollection
from graphgen.modules import GCN, GraphRCNN, Placeholder
from graphgen.utilities.factories import (
    DatasetCollection,
    DatasetFactory,
    PreprocessingFactory,
)
from graphgen.utilities.logging import log_metrics_stdout
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
    # Download and initialise resources
    print(colored("initialisation:", attrs=["bold"]))
    stanza.download(lang="en")

    # Print environment info
    print(colored("environment:", attrs=["bold"]))
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print(f"device: {torch.cuda.get_device_name(device) if cuda else 'CPU'}")
    print(config)

    if args.job == JobType.PREPROCESS:
        preprocess(config)
    elif args.job == JobType.TRAIN:
        run(config, device)
    elif args.job == JobType.TEST:
        raise NotImplementedError()
    else:
        raise NotImplementedError()


def preprocess(config: Config) -> None:
    """Preprocess `config.dataset` according to the `config.preprocessing` config."""
    print(colored("preprocessing:", attrs=["bold"]))
    factory = PreprocessingFactory()
    factory.process(config)


def run(config: Config, device: torch.device) -> None:
    """Train a model according to the `config.model` config."""
    print(colored("loading datasets:", attrs=["bold"]))
    factory = DatasetFactory()
    datasets, preprocessors = factory.create(config)
    print(f"train: {len(datasets.train)}")
    print(f"val: {len(datasets.val)}")
    print(f"test: {len(datasets.test)}")
    print(colored("model:", attrs=["bold"]))
    # TODO Use model factory
    # 1878 is the number of unique answers from the GQA paper
    # 1843 is the number of answers across train, val and testdev, returned by
    # len(preprocessors.questions.index_to_answer)
    model = Placeholder(
        GraphRCNN(len(preprocessors.scene_graphs.index_to_object)),
        GCN((300, 600, 900, 1200, 1500, len(preprocessors.questions.index_to_answer))),
    )
    model.to(device)
    model.train()
    print(f"{model=}")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.model.optimiser.learning_rate,
        weight_decay=config.model.optimiser.weight_decay,
    )
    print(f"{optimizer=}")
    criterion = torch.nn.NLLLoss()
    print(f"{criterion=}")

    # Run model
    print(colored("running:", attrs=["bold"]))
    train(model, criterion, optimizer, datasets, device, config)


def train(
    model: torch.nn.Module,
    criterion: Callable[..., torch.Tensor],
    optimizer: torch.optim.Optimizer,
    datasets: DatasetCollection,
    device: torch.device,
    config: Config,
) -> None:
    """Train a model on the data in `train_dataloader`, evaluating every epoch."""
    # pylint: disable=too-many-locals
    # Log gradients each epoch
    wandb.watch(
        model, log_freq=math.ceil(config.training.epochs / config.dataloader.batch_size)
    )
    dataloader = DataLoader(
        datasets.train,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.workers,
        sampler=ChunkedRandomSampler(datasets.train.questions),
        collate_fn=VariableSizeTensorCollator(),
    )
    metrics = MetricCollection(
        config, [Metric.ACCURACY, Metric.PRECISION, Metric.RECALL, Metric.F1]
    )
    for epoch in range(config.training.epochs):
        for batch, sample in enumerate(dataloader):
            # Move data to GPU
            deps = sample["question"]["dependencies"].to(device)
            targets = sample["question"]["answer"].to(device)
            images = [img.to(device) for img in sample["image"]]
            bbox_targets = [
                {"boxes": b.to(device), "labels": l.to(device)}
                for b, l in zip(
                    sample["scene_graph"]["boxes"], sample["scene_graph"]["labels"]
                )
            ]
            # Learn
            optimizer.zero_grad()
            dep_gcn_preds, grcnn_out = model(deps, images, bbox_targets)
            print(grcnn_out)
            print(dep_gcn_preds.size())
            print(targets.size())
            loss = criterion(dep_gcn_preds, targets)  # TODO incorporate grcnn loss
            loss.backward()
            optimizer.step()

            # Calculate and log metrics, using answer indices as we only want
            # basics for train set.
            metrics.append(
                sample["question"]["questionId"],
                np.argmax(dep_gcn_preds.detach().cpu().numpy(), axis=1),
                targets.detach().cpu().numpy(),
            )
            if (
                batch % config.training.log_step == config.training.log_step - 1
                or batch == len(dataloader) - 1
            ):
                results = {
                    "epoch": epoch + (batch + 1) / len(dataloader),
                    "train/loss": loss.item(),
                }
                results.update(
                    {f"train/{key}": val for key, val in metrics.evaluate().items()}
                )
                log_metrics_stdout(
                    results,
                    colors=(None, "yellow", "blue", "cyan", "cyan", "cyan"),
                    newline=False,
                )
                # Delay logging until after val metrics come in if end of epoch
                if batch != len(dataloader) - 1:
                    wandb.log(results)
                    metrics.reset()
        results.update(
            {
                f"val/{key}": val
                for key, val in evaluate(
                    model, criterion, datasets, metrics, device, config
                ).items()
            }
        )
        log_metrics_stdout(
            results,
            colors=(
                None,
                "yellow",
                "blue",
                "cyan",
                "cyan",
                "cyan",
                "yellow",
                "magenta",
                "magenta",
                "magenta",
                "magenta",
            ),
        )
        wandb.log(results)
        metrics.reset()


def evaluate(
    model: torch.nn.Module,
    criterion: Callable[..., torch.Tensor],
    datasets: DatasetCollection,
    metrics: MetricCollection,
    device: torch.device,
    config: Config,
) -> Dict[str, Any]:
    """Evaluate a model according to a criterion on a given dataset."""
    dataloader = DataLoader(
        datasets.val,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.workers,
        collate_fn=VariableSizeTensorCollator(),
    )
    metrics = MetricCollection(
        config, [Metric.ACCURACY, Metric.PRECISION, Metric.RECALL, Metric.F1]
    )
    model.eval()
    with torch.no_grad():
        for batch, sample in enumerate(dataloader):
            question = sample["question"]
            data = question["dependencies"].to(device)
            targets = question["answer"].to(device)
            preds = model(data)
            loss = criterion(preds, targets)
            # Calculate and log metrics, using answer indices as we only want
            # basics for train set.
            metrics.append(
                sample["question"]["questionId"],
                np.argmax(preds.detach().cpu().numpy(), axis=1),
                targets.detach().cpu().numpy(),
            )
            print(f"eval: {batch + 1}/{len(dataloader)}", end="\r")
    model.train()

    results = {"loss": loss.item()}
    results.update(metrics.evaluate())
    return results


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
    parser.add_argument(
        "--job",
        type=JobType,
        choices=list(iter(JobType)),
        metavar=str({str(job.value) for job in iter(JobType)}),
        help="Job type for this run.",
    )
    parser.add_argument(
        "--sync", action="store_true", help="Sync results to wandb if specified."
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
    # Parse config
    parsed_args = parse_args()
    config_obj, config_dict = load_config(parsed_args.config)
    # Set up wandb
    os.environ["WANDB_MODE"] = "run" if parsed_args.sync else "dryrun"
    if not Path(".wandb").exists():
        Path(".wandb").mkdir()
    wandb.init(
        project="graphgen",
        dir=".wandb",
        job_type=parsed_args.job.value,
        config=config_dict,
    )
    # Run main with parsed config
    main(parsed_args, config_obj)
