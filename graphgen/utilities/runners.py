"""Various train/val/test loops for running different types of models."""
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from ..config import Config
from ..datasets.collators import VariableSizeTensorCollator
from ..datasets.utilities import ChunkedRandomSampler
from ..metrics import Metric, MetricCollection
from ..utilities.factories import DatasetCollection
from .logging import log_metrics_stdout


@dataclass(frozen=True)
class ResumeInfo:
    """Information about the run being resumed."""

    run: str
    checkpoint: str


class Runner(ABC):
    """Abstract runner class for running models."""

    def __init__(
        self,
        config: Config,
        device: torch.device,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        criterion: Optional[Callable[..., torch.Tensor]],
        datasets: DatasetCollection,
        resume: Optional[ResumeInfo],
    ) -> None:
        """Create a Runner instance."""
        self.config = config
        self.device = device
        self.model = model
        self.optimiser = optimiser
        self.criterion = criterion
        self.datasets = datasets
        self.resume = resume

        self._start_epoch = 0

        if self.resume is not None:
            self._start_epoch = self.load()

    @abstractmethod
    def train(self) -> None:
        """Train a model according to a criterion and optimiser on a dataset."""

    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate a model according to a criterion on a given dataset."""

    def save(self, epoch: int, name: str) -> None:
        """Save a model's state dict to file."""
        # Set up weight checkpointing
        root = Path(wandb.run.dir) / "checkpoints"
        if not root.exists():
            root.mkdir(parents=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimiser.state_dict(),
            },
            str(root / name),
        )
        wandb.save(str(root / "*"))

    def load(self) -> int:
        """Load a model's state dict from file."""
        if self.resume is None:
            raise ValueError("Cannot load model without resume information.")
        root = Path(wandb.run.dir) / "checkpoints"
        if not root.exists():
            root.mkdir(parents=True)
        restored = wandb.restore(self.resume.checkpoint, run_path=self.resume.run)
        checkpoint = torch.load(restored.name)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optimizer_state_dict"])

        epoch: int = checkpoint["epoch"]

        print(f"Loaded state dict from {restored.name}.")
        return epoch


class FasterRCNNRunner(Runner):
    """Runner class for training a FasterRCNN model."""

    def train(self) -> None:
        """Train a model according to a criterion and optimiser on a dataset."""
        # Log gradients each epoch
        wandb.watch(
            self.model,
            log_freq=math.ceil(
                self.config.training.epochs / self.config.training.dataloader.batch_size
            ),
        )

        # Create dataloader
        dataloader = DataLoader(
            self.datasets.train,
            batch_size=self.config.training.dataloader.batch_size,
            num_workers=self.config.training.dataloader.workers,
            sampler=ChunkedRandomSampler(self.datasets.train.questions),
            collate_fn=VariableSizeTensorCollator(),
        )

        if self._start_epoch == 0:
            self.save(self._start_epoch, "0.pt")

        self.model.train()
        self.model.to(self.device)

        for epoch in range(self._start_epoch, self.config.training.epochs):
            for batch, sample in enumerate(dataloader):
                # Move data to GPU
                images = [img.to(self.device) for img in sample["image"]]
                targets = [
                    {"boxes": b.to(self.device), "labels": l.to(self.device)}
                    for b, l in zip(
                        sample["scene_graph"]["boxes"], sample["scene_graph"]["labels"]
                    )
                ]
                print(images)
                print(targets)

                # Learn
                self.optimiser.zero_grad()
                output = self.model(images=images, targets=targets)
                loss: torch.Tensor = sum(list(output.values()))
                loss.backward()
                self.optimiser.step()

                # Log metrics and loss if we are at a logging batch
                if (
                    batch % self.config.training.log_step
                    == self.config.training.log_step - 1
                    or batch == len(dataloader) - 1
                ):
                    results = {
                        "epoch": epoch + (batch + 1) / len(dataloader),
                        "train/loss": loss.item(),
                        "train/roi-classifier/loss": output["loss_classifier"].item(),
                        "train/roi-regression/loss": output["loss_box_reg"].item(),
                        "train/rpn-objectness/loss": output["loss_objectness"].item(),
                        "train/rpn-regression/loss": output["loss_rpn_box_reg"].item(),
                    }
                    log_metrics_stdout(results, newline=False)

                    # Delay wandb logging until after val metrics come in if at
                    # end of epoch to avoid duped logs
                    if batch != len(dataloader) - 1:
                        wandb.log(results)

            self.save(epoch + 1, f"{epoch+1}.pt")
            results.update({f"val/{key}": val for key, val in self.evaluate().items()})
            log_metrics_stdout(results)
            wandb.log(results)

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate a model according to a criterion on a given dataset."""
        dataloader = DataLoader(
            self.datasets.val,
            batch_size=self.config.training.dataloader.batch_size,
            num_workers=self.config.training.dataloader.workers,
            collate_fn=VariableSizeTensorCollator(),
        )
        self.model.eval()

        eval_limit = self.config.training.eval_subset

        with torch.no_grad():
            for batch, sample in enumerate(dataloader):
                # Move data to GPU
                images = [img.to(self.device) for img in sample["image"]]
                targets = [
                    {"boxes": b.to(self.device), "labels": l.to(self.device)}
                    for b, l in zip(
                        sample["scene_graph"]["boxes"], sample["scene_graph"]["labels"]
                    )
                ]
                _ = self.model(images=images, targets=targets)

                # TODO add bbox logging

                print(f"eval: {batch + 1}/{eval_limit}", end="\r")
                if batch + 1 == eval_limit:
                    break

        self.model.train()
        results: Dict[str, Any] = {}
        return results


class EndToEndMultiChannelGCNRunner(Runner):
    """Runner class for training a MultiChannelGCN model."""

    def __init__(
        self,
        config: Config,
        device: torch.device,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        criterion: Optional[Callable[..., torch.Tensor]],
        datasets: DatasetCollection,
        resume: Optional[ResumeInfo],
    ) -> None:
        """Create a MultiChannelGCNRunner instance."""
        if criterion is None:
            raise ValueError("This model requires a criterion.")
        super().__init__(config, device, model, optimiser, criterion, datasets, resume)

    def train(self) -> None:
        """Train a model according to a criterion and optimiser on a dataset."""
        # Log gradients each epoch
        wandb.watch(
            self.model,
            log_freq=math.ceil(
                self.config.training.epochs / self.config.training.dataloader.batch_size
            ),
        )
        dataloader = DataLoader(
            self.datasets.train,
            batch_size=self.config.training.dataloader.batch_size,
            num_workers=self.config.training.dataloader.workers,
            sampler=ChunkedRandomSampler(self.datasets.train.questions),
            collate_fn=VariableSizeTensorCollator(),
        )
        metrics = MetricCollection(
            self.config, [Metric.ACCURACY, Metric.PRECISION, Metric.RECALL, Metric.F1]
        )

        if self._start_epoch == 0:
            self.save(self._start_epoch, "0.pt")

        self.model.train()
        self.model.to(self.device)

        for epoch in range(self._start_epoch, self.config.training.epochs):
            for batch, sample in enumerate(dataloader):
                # Move data to GPU
                deps = sample["question"]["dependencies"].to(self.device)
                targets = sample["question"]["answer"].to(self.device)
                images = [img.to(self.device) for img in sample["image"]]
                bbox_targets = [
                    {"boxes": b.to(self.device), "labels": l.to(self.device)}
                    for b, l in zip(
                        sample["scene_graph"]["boxes"], sample["scene_graph"]["labels"]
                    )
                ]
                # Learn
                self.optimiser.zero_grad()
                rcnn_loss, preds = self.model(
                    dependencies=deps, images=images, targets=bbox_targets
                )
                pred_loss = self.criterion(preds, targets)  # type: ignore

                # Compute multi-task loss
                loss = pred_loss
                for partial_loss in rcnn_loss.values():
                    loss += partial_loss
                loss.backward()
                self.optimiser.step()

                # Calculate and log metrics, using answer indices as we only want
                # basics for train set.
                metrics.append(
                    sample["question"]["questionId"],
                    np.argmax(preds.detach().cpu().numpy(), axis=1),
                    targets.detach().cpu().numpy(),
                )
                if (
                    batch % self.config.training.log_step
                    == self.config.training.log_step - 1
                    or batch == len(dataloader) - 1
                ):
                    results = {
                        "epoch": epoch + (batch + 1) / len(dataloader),
                        "train/loss": pred_loss.item(),
                        "train/roi-classifier/loss": rcnn_loss[
                            "loss_classifier"
                        ].item(),
                        "train/roi-regression/loss": rcnn_loss["loss_box_reg"].item(),
                        "train/rpn-objectness/loss": rcnn_loss[
                            "loss_objectness"
                        ].item(),
                        "train/rpn-regression/loss": rcnn_loss[
                            "loss_rpn_box_reg"
                        ].item(),
                    }

                    results.update(
                        {f"train/{key}": val for key, val in metrics.evaluate().items()}
                    )
                    log_metrics_stdout(
                        results,
                        newline=False,
                    )
                    # Delay logging until after val metrics come in if end of epoch
                    if batch != len(dataloader) - 1:
                        wandb.log(results)
                        metrics.reset()
            # Save and log at end of epoch
            self.save(epoch + 1, f"{epoch+1}.pt")
            results.update({f"val/{key}": val for key, val in self.evaluate().items()})
            log_metrics_stdout(results)
            wandb.log(results)
            metrics.reset()

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate a model according to a criterion on a given dataset."""
        dataloader = DataLoader(
            self.datasets.val,
            batch_size=self.config.training.dataloader.batch_size,
            num_workers=self.config.training.dataloader.workers,
            collate_fn=VariableSizeTensorCollator(),
        )
        metrics = MetricCollection(
            self.config, [Metric.ACCURACY, Metric.PRECISION, Metric.RECALL, Metric.F1]
        )
        self.model.eval()

        eval_limit = self.config.training.eval_subset

        with torch.no_grad():
            for batch, sample in enumerate(dataloader):
                # Move data to GPU
                deps = sample["question"]["dependencies"].to(self.device)
                targets = sample["question"]["answer"].to(self.device)
                images = [img.to(self.device) for img in sample["image"]]
                bbox_targets = [
                    {"boxes": b.to(self.device), "labels": l.to(self.device)}
                    for b, l in zip(
                        sample["scene_graph"]["boxes"], sample["scene_graph"]["labels"]
                    )
                ]
                _, preds = self.model(  # First sample is rcnn_preds
                    dependencies=deps, images=images, targets=bbox_targets
                )
                loss = self.criterion(preds, targets)  # type: ignore

                # TODO add bbox logging

                # Calculate and log metrics, using answer indices as we only want
                # basics for train set.
                metrics.append(
                    sample["question"]["questionId"],
                    np.argmax(preds.detach().cpu().numpy(), axis=1),
                    targets.detach().cpu().numpy(),
                )
                print(f"eval: {batch + 1}/{eval_limit}", end="\r")
                if batch + 1 == eval_limit:
                    break

        self.model.train()
        results = {"loss": loss.item()}
        results.update(metrics.evaluate())
        return results