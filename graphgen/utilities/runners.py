"""Various train/val/test loops for running different types of models."""
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from termcolor import colored
from torch.utils.data import DataLoader

from ..config import Config
from ..datasets.collators import VariableSizeTensorCollator
from ..datasets.utilities.chunked_random_sampler import ChunkedRandomSampler
from ..metrics import Metric, MetricCollection
from ..utilities.preprocessing import DatasetCollection, PreprocessorCollection
from .logging import log_metrics_stdout


@dataclass
class ResumeInfo:
    """Information about the run being resumed."""

    run: str
    checkpoint: str


class Runner(ABC):
    """Abstract runner class for running models."""

    # pylint: disable=too-many-instance-attributes

    def __init__(  # pylint: disable=too-many-arguments
        self,
        config: Config,
        device: torch.device,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        criterion: Optional[Callable[..., torch.Tensor]],
        datasets: DatasetCollection,
        preprocessors: PreprocessorCollection,
        resume: Optional[ResumeInfo],
    ) -> None:
        """Create a Runner instance."""
        self.config = config
        self.device = device
        self.model = model
        self.optimiser = optimiser
        self.criterion = criterion
        self.datasets = datasets
        self.preprocessors = preprocessors
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
        print(colored("loading checkpoint:", attrs=["bold"]))
        if self.resume is None:
            raise ValueError("Cannot load model without resume information.")
        root = Path(wandb.run.dir) / "checkpoints"
        if not root.exists():
            root.mkdir(parents=True)
        restored = wandb.restore(
            self.resume.checkpoint, run_path=self.resume.run, root=root
        )
        checkpoint = torch.load(restored.name)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optimizer_state_dict"])

        epoch: int = checkpoint["epoch"]

        print(f"loaded checkpoint from {restored.name}")
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

        # Ensure we are using the whole training set
        assert abs(self.config.training.data.train.subset[0] - 0.0) < 1e-16
        assert abs(self.config.training.data.train.subset[1] - 1.0) < 1e-16

        # Create dataloader
        dataloader = DataLoader(
            self.datasets.train,
            batch_size=self.config.training.dataloader.batch_size,
            num_workers=self.config.training.dataloader.workers,
            shuffle=True,
            collate_fn=VariableSizeTensorCollator(),
        )

        if self._start_epoch == 0:
            self.save(self._start_epoch, "0.pt")

        self.model.train()
        self.model.to(self.device)

        for epoch in range(self._start_epoch, self.config.training.epochs):
            for batch, sample in enumerate(dataloader):
                # Move data to GPU, filtering if there are no bounding boxes,
                # as GQA bboxes with zero width or height were removed in
                # preprocessing step.
                data = [
                    (
                        img.to(self.device),
                        {
                            "boxes": boxes.to(self.device),
                            "labels": labels.to(self.device),
                        },
                    )
                    for img, boxes, labels in zip(
                        sample["image"],
                        sample["scene_graph"]["boxes"],
                        sample["scene_graph"]["labels"],
                    )
                    if len(boxes) >= 1 and len(labels) >= 1
                ]
                if len(data) == 0:
                    continue

                images, targets = list(zip(*data))

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
            log_metrics_stdout(results)
            bbox_images = self.evaluate()
            wandb.log(results.update(bbox_images))

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate a model according to a criterion on a given dataset."""
        # pylint: disable=too-many-locals
        start = int(self.config.training.data.val.subset[0] * len(self.datasets.val))
        end = int(self.config.training.data.val.subset[1] * len(self.datasets.val))
        dataloader = DataLoader(
            torch.utils.data.Subset(self.datasets.val, range(start, end)),
            batch_size=self.config.training.dataloader.batch_size,
            num_workers=self.config.training.dataloader.workers,
            collate_fn=VariableSizeTensorCollator(),
        )
        self.model.eval()

        image_sample_limit = 32
        all_images: List[wandb.Image] = []
        obj_classes = {
            idx: obj
            for obj, idx in self.preprocessors.scene_graphs.object_to_index.items()
        }
        with torch.no_grad():
            for batch, sample in enumerate(dataloader):
                # Move data to GPU
                data = [
                    (
                        img.to(self.device),
                        {
                            "boxes": boxes.to(self.device),
                            "labels": labels.to(self.device),
                        },
                    )
                    for img, boxes, labels in zip(
                        sample["image"],
                        sample["scene_graph"]["boxes"],
                        sample["scene_graph"]["labels"],
                    )
                    if len(boxes) >= 1 and len(labels) >= 1
                ]

                if len(data) == 0:
                    continue

                images, targets = list(zip(*data))

                preds = self.model(images=images, targets=targets)

                for pred, image, target in zip(preds, images, targets):
                    if len(all_images) >= image_sample_limit:
                        break
                    log_img = wandb.Image(
                        image,
                        boxes={
                            "predictions": {
                                "box_data": [
                                    {
                                        "position": {
                                            "minX": box[0].item(),
                                            "maxX": box[2].item(),
                                            "minY": box[1].item(),
                                            "maxY": box[3].item(),
                                        },
                                        "class_id": lbl.item(),
                                        "box_caption": f"{obj_classes[lbl.item()]} ({score.item():.2f})",  # noqa: B950
                                        "domain": "pixel",
                                        "scores": {"score": score.item()},
                                    }
                                    for box, lbl, score in zip(
                                        pred["boxes"],
                                        pred["labels"],
                                        pred["scores"],
                                    )
                                ],
                                "class_labels": obj_classes,
                            },
                            "ground_truth": {
                                "box_data": [
                                    {
                                        "position": {
                                            "minX": box[0].item(),
                                            "maxX": box[2].item(),
                                            "minY": box[1].item(),
                                            "maxY": box[3].item(),
                                        },
                                        "class_id": lbl.item(),
                                        "box_caption": obj_classes[lbl.item()],
                                        "domain": "pixel",
                                    }
                                    for box, lbl in zip(
                                        target["boxes"], target["labels"]
                                    )
                                ],
                                "class_labels": obj_classes,
                            },
                        },
                    )
                    all_images.append(log_img)

                print(f"eval: {batch + 1}/{len(dataloader)}", end="\r")

        self.model.train()
        return {"boxes": all_images}


class EndToEndMultiChannelGCNRunner(Runner):
    """Runner class for training a MultiChannelGCN model with built in FasterRCNN \
    object detector."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        config: Config,
        device: torch.device,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        criterion: Optional[Callable[..., torch.Tensor]],
        datasets: DatasetCollection,
        preprocessors: PreprocessorCollection,
        resume: Optional[ResumeInfo],
    ) -> None:
        """Create a MultiChannelGCNRunner instance."""
        if criterion is None:
            raise ValueError("This model requires a criterion.")
        super().__init__(
            config, device, model, optimiser, criterion, datasets, preprocessors, resume
        )

    def train(self) -> None:
        """Train a model according to a criterion and optimiser on a dataset."""
        # Log gradients each epoch
        wandb.watch(
            self.model,
            log_freq=math.ceil(
                self.config.training.epochs / self.config.training.dataloader.batch_size
            ),
        )

        # Ensure we are using the whole training set
        assert abs(self.config.training.data.train.subset[0] - 0.0) < 1e-16
        assert abs(self.config.training.data.train.subset[1] - 1.0) < 1e-16

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
        start = int(self.config.training.data.val.subset[0] * len(self.datasets.val))
        end = int(self.config.training.data.val.subset[1] * len(self.datasets.val))
        dataloader = DataLoader(
            torch.utils.data.Subset(self.datasets.val, range(start, end)),
            batch_size=self.config.training.dataloader.batch_size,
            num_workers=self.config.training.dataloader.workers,
            collate_fn=VariableSizeTensorCollator(),
        )
        metrics = MetricCollection(
            self.config, [Metric.ACCURACY, Metric.PRECISION, Metric.RECALL, Metric.F1]
        )
        self.model.eval()

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
                print(f"eval: {batch + 1}/{len(dataloader)}", end="\r")

        self.model.train()
        results = {"loss": loss.item()}
        results.update(metrics.evaluate())
        return results


class MultiChannelGCNRunner(Runner):
    """Runner class for training a MultiChannelGCN model."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        config: Config,
        device: torch.device,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        criterion: Optional[Callable[..., torch.Tensor]],
        datasets: DatasetCollection,
        preprocessors: PreprocessorCollection,
        resume: Optional[ResumeInfo],
    ) -> None:
        """Create a MultiChannelGCNRunner instance."""
        if criterion is None:
            raise ValueError("This model requires a criterion.")
        super().__init__(
            config, device, model, optimiser, criterion, datasets, preprocessors, resume
        )

    def train(self) -> None:
        """Train a model according to a criterion and optimiser on a dataset."""
        # Log gradients each epoch
        wandb.watch(
            self.model,
            log_freq=math.ceil(
                self.config.training.epochs / self.config.training.dataloader.batch_size
            ),
        )

        # Ensure we are using the whole training set
        assert abs(self.config.training.data.train.subset[0] - 0.0) < 1e-16
        assert abs(self.config.training.data.train.subset[1] - 1.0) < 1e-16

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
            self.save(self._start_epoch, "current.pt")

        self.model.train()
        self.model.to(self.device)

        best_val_loss = math.inf
        for epoch in range(self._start_epoch, self.config.training.epochs):
            for batch, sample in enumerate(dataloader):
                # Move data to GPU
                dependencies = sample["question"]["dependencies"].to(self.device)
                # word_embeddings = pack_sequence(
                #     sample["question"]["embeddings"], enforce_sorted=False
                # ).to(self.device)
                objects = sample["scene_graph"]["objects"].to(self.device)
                targets = sample["question"]["answer"].to(self.device)

                # Labels can be indices or a object class probability distribution.

                # Learn
                self.optimiser.zero_grad()
                preds = self.model(dependencies=dependencies, objects=objects)
                loss = self.criterion(preds, targets)  # type: ignore

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
                        "train/loss": loss.item(),
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
            self.save(epoch + 1, "current.pt")

            results.update({f"val/{key}": val for key, val in self.evaluate().items()})
            if results["val/loss"] <= best_val_loss:
                best_val_loss = results["val/loss"]
                self.save(epoch + 1, "best.pt")
            log_metrics_stdout(results)
            wandb.log(results)
            metrics.reset()

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate a model according to a criterion on a given dataset."""
        start = int(self.config.training.data.val.subset[0] * len(self.datasets.val))
        end = int(self.config.training.data.val.subset[1] * len(self.datasets.val))
        dataloader = DataLoader(
            torch.utils.data.Subset(self.datasets.val, range(start, end)),
            batch_size=self.config.training.dataloader.batch_size,
            num_workers=self.config.training.dataloader.workers,
            collate_fn=VariableSizeTensorCollator(),
        )
        metrics = MetricCollection(
            self.config, [Metric.ACCURACY, Metric.PRECISION, Metric.RECALL, Metric.F1]
        )
        self.model.eval()

        with torch.no_grad():
            for batch, sample in enumerate(dataloader):
                # Move data to GPU
                dependencies = sample["question"]["dependencies"].to(self.device)
                # word_embeddings = pack_sequence(
                #     sample["question"]["embeddings"], enforce_sorted=False
                # ).to(self.device)
                objects = sample["scene_graph"]["objects"].to(self.device)
                targets = sample["question"]["answer"].to(self.device)

                # Labels can be indices or a object class probability distribution.

                # Learn
                preds = self.model(dependencies=dependencies, objects=objects)
                loss = self.criterion(preds, targets)  # type: ignore

                # Calculate and log metrics, using answer indices as we only want
                # basics for train set.
                metrics.append(
                    sample["question"]["questionId"],
                    np.argmax(preds.detach().cpu().numpy(), axis=1),
                    targets.detach().cpu().numpy(),
                )
                print(f"eval: {batch + 1}/{len(dataloader)}", end="\r")

        self.model.train()
        results = {"loss": loss.item()}
        results.update(metrics.evaluate())
        return results


class VQAModelRunner(Runner):
    """Runner class for training a VQA model."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        config: Config,
        device: torch.device,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        criterion: Optional[Callable[..., torch.Tensor]],
        datasets: DatasetCollection,
        preprocessors: PreprocessorCollection,
        resume: Optional[ResumeInfo],
    ) -> None:
        """Create a `VQAModelRunner` instance."""
        if criterion is None:
            raise ValueError("This model requires a criterion.")
        super().__init__(
            config, device, model, optimiser, criterion, datasets, preprocessors, resume
        )

        self.scheduler = None
        if self.config.training.optimiser.schedule:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimiser,
                "min",
                factor=0.5,
                patience=1,
            )

    def train(self) -> None:
        """Train a model according to a criterion and optimiser on a dataset."""
        # Log gradients each epoch
        wandb.watch(
            self.model,
            log_freq=math.ceil(
                self.config.training.epochs / self.config.training.dataloader.batch_size
            ),
        )

        # Ensure we are using the whole training set
        assert abs(self.config.training.data.train.subset[0] - 0.0) < 1e-16
        assert abs(self.config.training.data.train.subset[1] - 1.0) < 1e-16

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
            self.save(self._start_epoch, "current.pt")

        self.model.train()
        self.model.to(self.device)

        best_val_loss = math.inf
        for epoch in range(self._start_epoch, self.config.training.epochs):
            for batch, sample in enumerate(dataloader):
                # Move data to GPU
                dependencies = sample["question"]["dependencies"].to(self.device)
                objects = sample["scene_graph"]["objects"].to(self.device)
                targets = sample["question"]["answer"].to(self.device)

                self.optimiser.zero_grad()
                preds = F.log_softmax(
                    self.model(dependencies=dependencies, objects=objects), dim=1
                )
                loss = self.criterion(preds, targets)  # type: ignore
                loss.backward()

                if self.config.training.optimiser.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.training.optimiser.grad_clip,
                    )

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
                        "train/loss": loss.item(),
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
            self.save(epoch + 1, "current.pt")

            results.update({f"val/{key}": val for key, val in self.evaluate().items()})
            if results["val/loss"] <= best_val_loss:
                best_val_loss = results["val/loss"]
                self.save(epoch + 1, "best.pt")
            log_metrics_stdout(results)
            wandb.log(results)
            metrics.reset()

            # Update lr based on val loss (official MAC code uses train loss)
            if self.scheduler is not None:
                self.scheduler.step(results["val/loss"])

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate a model according to a criterion on a given dataset."""
        start = int(self.config.training.data.val.subset[0] * len(self.datasets.val))
        end = int(self.config.training.data.val.subset[1] * len(self.datasets.val))
        dataloader = DataLoader(
            torch.utils.data.Subset(self.datasets.val, range(start, end)),
            batch_size=self.config.training.dataloader.batch_size,
            num_workers=self.config.training.dataloader.workers,
            collate_fn=VariableSizeTensorCollator(),
        )
        metrics = MetricCollection(
            self.config, [Metric.ACCURACY, Metric.PRECISION, Metric.RECALL, Metric.F1]
        )

        self.model.eval()

        with torch.no_grad():
            for batch, sample in enumerate(dataloader):
                # Move data to GPU
                dependencies = sample["question"]["dependencies"].to(self.device)
                objects = sample["scene_graph"]["objects"].to(self.device)
                targets = sample["question"]["answer"].to(self.device)

                # Learn
                preds = F.log_softmax(
                    self.model(dependencies=dependencies, objects=objects), dim=1
                )
                loss = self.criterion(preds, targets)  # type: ignore

                # Calculate and log metrics, using answer indices as we only want
                # basics for train set.
                metrics.append(
                    sample["question"]["questionId"],
                    np.argmax(preds.detach().cpu().numpy(), axis=1),
                    targets.detach().cpu().numpy(),
                )
                print(f"eval: {batch + 1}/{len(dataloader)}", end="\r")

        self.model.train()
        results = {"loss": loss.item()}
        results.update(metrics.evaluate())
        return results
