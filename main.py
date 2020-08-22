"""Main entrypoint for running, loading, saving and evaluating models."""

import argparse
import json
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
from torch_geometric.data import DataLoader

from graphgen.config import Config
from graphgen.datasets.factory import DatasetFactory, PreprocessingFactory
from graphgen.datasets.utilities import ChunkedRandomSampler
from graphgen.modules.gcn import GCN
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

    if args.job == JobType.PREPROCESS:
        preprocess(config)
    elif args.job == JobType.TRAIN:
        print(colored("loading datasets:", attrs=["bold"]))
        factory = DatasetFactory()
        train_data, val_data, test_data = factory.create(config)

        print(colored("model:", attrs=["bold"]))
        model = GCN((300, 600, 1200, 1878))  # 1878 is number of unique answers
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
        sampler = ChunkedRandomSampler(train_data.questions)
        train_dataloader = DataLoader(
            train_data,
            batch_size=config.dataloader.batch_size,
            num_workers=config.dataloader.workers,
            sampler=sampler,
        )
        val_dataloader = DataLoader(
            val_data,
            batch_size=config.dataloader.batch_size,
            num_workers=config.dataloader.workers,
        )
        train(
            model,
            criterion,
            optimizer,
            train_dataloader,
            val_dataloader,
            device,
            config,
        )
    elif args.job == JobType.TEST:
        raise NotImplementedError()
    else:
        raise NotImplementedError()


def preprocess(config: Config) -> None:
    """Preprocess `config.dataset` according to the `config.preprocessing` config."""
    print(colored("preprocessing:", attrs=["bold"]))
    factory = PreprocessingFactory()
    factory.process(config)


def train(
    model: torch.nn.Module,
    criterion: Callable[..., torch.Tensor],
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Config,
) -> None:
    """Train a model on the data in `train_dataloader`, evaluating every epoch."""
    # pylint: disable=too-many-locals  # TODO abstract metric calculation logic.
    for epoch in range(config.training.epochs):
        metrics: Dict[str, Any] = {}
        correct = 0
        total = 0
        for batch, sample in enumerate(train_dataloader):
            # Move data to GPU
            data = sample["question"]["dependencies"].to(device)
            targets = sample["question"]["answer"].to(device)

            # Learn
            optimizer.zero_grad()
            preds = model(data)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            # Calculate and log metrics
            preds_np = np.argmax(preds.detach().cpu().numpy(), axis=1)
            targets_np = targets.detach().cpu().numpy()
            correct += np.sum(np.equal(preds_np, targets_np))
            total += len(targets_np)

            if (
                batch % config.training.log_step == config.training.log_step - 1
                or batch == len(train_dataloader) - 1
            ):
                metrics = {
                    "epoch": epoch + (batch + 1) / len(train_dataloader),
                    "train/loss": loss.item(),
                    "train/accuracy": correct / total,
                }
                log_metrics_stdout(
                    metrics, colors=(None, "cyan", "cyan"), newline=False,
                )
                wandb.log(metrics)
        val_metrics = evaluate(model, criterion, val_dataloader, device)
        metrics.update({f"val/{key}": val for key, val in val_metrics.items()})

        log_metrics_stdout(metrics, colors=(None, "cyan", "cyan", "magenta", "magenta"))
        wandb.log(metrics)


def evaluate(
    model: torch.nn.Module,
    criterion: Callable[..., torch.Tensor],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate a model according to a criterion on a given dataset."""
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch, sample in enumerate(dataloader):
            question = sample["question"]
            data = question["dependencies"].to(device)
            targets = question["answer"].to(device)
            preds = model(data)
            loss = criterion(preds, targets)
            preds_np = np.argmax(preds.detach().cpu().numpy(), axis=1)
            targets_np = targets.detach().cpu().numpy()
            correct += np.sum(np.equal(preds_np, targets_np))
            total += len(targets_np)
            print(f"eval: {batch + 1}/{len(dataloader)}", end="\r")

    model.train()
    return {"loss": loss.item(), "accuracy": correct / total}


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
