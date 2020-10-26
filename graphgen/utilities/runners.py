"""Various train/val/test loops for running different types of models."""
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from termcolor import colored
from torch.utils.data import DataLoader
from torch_geometric.nn.conv import GATConv

from ..config import Config
from ..datasets.collators import VariableSizeTensorCollator
from ..datasets.utilities.chunked_random_sampler import ChunkedRandomSampler
from ..metrics import Metric, MetricCollection
from ..schemas.common import BoundingBox
from ..utilities.preprocessing import DatasetCollection, PreprocessorCollection
from .hooks import GATConvAttentionHook
from .logging import log_metrics_stdout
from .visualisation import SparseGraphVisualiser, plot_image


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

    @abstractmethod
    def visualise(self, count: int = 1) -> Dict[str, Any]:
        """Visualise samples from the validation set."""

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
        wandb.save(str(root / "*"), wandb.run.dir)

    def load(self) -> int:
        """Load a model's state dict from file."""
        print(colored("loading checkpoint:", attrs=["bold"]))
        if self.resume is None:
            raise ValueError("Cannot load model without resume information.")
        root = Path(wandb.run.dir)
        if not root.exists():
            root.mkdir(parents=True)
        restored = wandb.restore(
            self.resume.checkpoint, run_path=self.resume.run, root=root
        )
        checkpoint = torch.load(restored.name)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in self.optimiser.state.values():
            for key, val in state.items():
                if isinstance(val, torch.Tensor):
                    state[key] = val.to(self.device)
        epoch: int = checkpoint["epoch"]

        print(f"loaded checkpoint from {restored.name}")
        return epoch


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
            metrics=[Metric.ACCURACY, Metric.PRECISION, Metric.RECALL, Metric.F1],
            labels=self.preprocessors.questions.index_to_answer,
        )

        self.model.train()
        self.model.to(self.device)

        #  TEMPORARY
        vis_count = 32
        best_val_loss = math.inf
        if self._start_epoch == 0:
            # Start with val to offset hyperband indices
            self.save(self._start_epoch, "current.pt")
            self.save(self._start_epoch, "best.pt")
            results = {f"val/{key}": val for key, val in self.evaluate().items()}
            best_val_loss = results["val/loss"]
            log_metrics_stdout(results)
            results.update(
                {f"vis/{key}": val for key, val in self.visualise(vis_count).items()}
            )
            wandb.log(results)
        else:
            # TEMPORARY
            results = {
                f"vis/{key}": val for key, val in self.visualise(vis_count).items()
            }
            wandb.log(results)

        for epoch in range(self._start_epoch, self.config.training.epochs):
            for batch, sample in enumerate(dataloader):
                # Move data to GPU
                dependencies = sample["question"]["dependencies"].to(self.device)
                graph = sample["scene_graph"]["graph"].to(self.device)
                targets = sample["question"]["answer"].to(self.device)

                self.optimiser.zero_grad()
                preds = F.log_softmax(
                    self.model(question_graph=dependencies, scene_graph=graph), dim=1
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
            results.update(
                {f"vis/{key}": val[0] for key, val in self.visualise().items()}
            )
            wandb.log(results)
            metrics.reset()

            # Update lr based on val loss (official MAC code uses train loss)
            if self.scheduler is not None:
                self.scheduler.step(results["val/loss"])

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate a model according to a criterion on a given dataset."""
        # Determine dataset bounds
        start = int(self.config.training.data.val.subset[0] * len(self.datasets.val))
        end = int(self.config.training.data.val.subset[1] * len(self.datasets.val))

        # Prepare for evaluation
        self.model.eval()
        train_reduction = self.criterion.reduction  # type: ignore
        self.criterion.reduction = "sum"  # type: ignore
        loss = 0

        # Evaluate metrics
        metrics = MetricCollection(
            metrics=[Metric.ACCURACY, Metric.PRECISION, Metric.RECALL, Metric.F1],
            labels=self.preprocessors.questions.index_to_answer,
        )
        dataloader = DataLoader(
            torch.utils.data.Subset(self.datasets.val, range(start, end)),
            batch_size=self.config.training.dataloader.batch_size,
            num_workers=self.config.training.dataloader.workers,
            collate_fn=VariableSizeTensorCollator(),
        )
        with torch.no_grad():
            for batch, sample in enumerate(dataloader):
                # Move data to GPU
                dependencies = sample["question"]["dependencies"].to(self.device)
                graph = sample["scene_graph"]["graph"].to(self.device)
                targets = sample["question"]["answer"].to(self.device)

                # Learn
                preds = F.log_softmax(
                    self.model(question_graph=dependencies, scene_graph=graph), dim=1
                )
                loss += self.criterion(preds, targets).item()  # type: ignore

                # Calculate and log metrics, using answer indices as we only want
                # basics for train set.
                metrics.append(
                    sample["question"]["questionId"],
                    np.argmax(preds.detach().cpu().numpy(), axis=1),
                    targets.detach().cpu().numpy(),
                )
                print(f"eval: {batch + 1}/{len(dataloader)}", end="\r")

        # Reset model to train, reset criterion reduction ready for training
        self.model.train()
        self.criterion.reduction = train_reduction  # type: ignore

        # Report average loss across all val samples
        results = {"loss": loss / len(dataloader.dataset)}

        # Include metrics like accuracy, precision etc. from MetricCollection
        results.update(metrics.evaluate())

        return results

    def visualise(
        self, sample_limit: int = 1, row_limit: int = 10000
    ) -> Dict[str, Any]:
        """Visualise a fixed number of samples to view the model's reasoning."""
        # Determine dataset bounds
        start = int(self.config.training.data.val.subset[0] * len(self.datasets.val))
        end = int(self.config.training.data.val.subset[1] * len(self.datasets.val))
        end = min(sample_limit, end)

        # Prepare for evaluation
        self.model.eval()

        visualisations: Dict[str, Any] = {"images": []}
        handles = []

        # Gather attention maps with forward hooks
        gat_hooks = {}
        for name, module in self.model.named_modules():
            if isinstance(module, GATConv):
                gat_hooks[name] = GATConvAttentionHook()
                handle = module.register_forward_hook(gat_hooks[name])
                handles.append(handle)

        dataloader = DataLoader(
            torch.utils.data.Subset(self.datasets.val, range(start, end)),
            batch_size=1,
            num_workers=0,
            collate_fn=VariableSizeTensorCollator(),
        )
        gat_graph_visualiser = SparseGraphVisualiser()
        with torch.no_grad():
            for idx, sample in enumerate(dataloader):
                # Load data
                dependencies = sample["question"]["dependencies"].to(self.device)
                graph = sample["scene_graph"]["graph"].to(self.device)
                image_id = sample["scene_graph"]["imageId"][0]
                image = self.datasets.images[
                    self.datasets.images.key_to_index(image_id)
                ]
                boxes = {
                    "ground_truth": [
                        BoundingBox(box[0], box[1], box[2], box[3], label=lbl)
                        for box, lbl in zip(
                            sample["scene_graph"]["boxes"][0].tolist(),
                            [
                                self.preprocessors.scene_graphs.object_to_index[
                                    obj_lbl[0]
                                ]
                                for obj_lbl in sample["scene_graph"]["labels"]
                            ],
                        )
                    ]
                }
                labels = [obj_lbl[0] for obj_lbl in sample["scene_graph"]["labels"]]
                relations = [
                    rel_lbl[0] for rel_lbl in sample["scene_graph"]["relations"]
                ]
                attributes = [
                    attr_lbl[0]
                    for obj_attrs in sample["scene_graph"]["attributes"]
                    for attr_lbl in obj_attrs
                ]
                # Remove all duplicate attributes since attribute nodes are shared
                unique_attributes = []
                for attr in attributes:
                    if attr not in unique_attributes:
                        unique_attributes.append(attr)

                # Propagate data forward
                self.model(question_graph=dependencies, scene_graph=graph)

                # Add images with labeled ground truth bounding boxes

                visualisations["images"].append(
                    plot_image(
                        image,
                        caption=image_id,
                        boxes=boxes,
                    )
                )

                # wandb bounding boxes are bugged:
                # https://github.com/wandb/client/issues/1348
                # visualisations["images"].append(wandb_image(
                #     image,
                #     caption=image_id,
                #     boxes=boxes,
                #     object_to_index=self.preprocessors.scene_graphs.object_to_index,
                # ))

                # Add scene graoh nodes and edges to a table
                node_labels = labels + relations + unique_attributes
                node_types = (
                    ["object"] * len(labels)
                    + ["relation"] * len(relations)
                    + ["attribute"] * len(unique_attributes)
                )
                edge_index = None
                values = {}
                for name, hook in gat_hooks.items():
                    if hook.result is not None:
                        indices, attn_weights = hook.result
                        if edge_index is not None:
                            assert torch.allclose(indices, edge_index)
                        edge_index = indices
                        values[name] = attn_weights
                try:
                    gat_graph_visualiser.add_graph(
                        node_labels, node_types, edge_index, values
                    )
                except ValueError:
                    break

        visualisations["scene_graph_nodes"] = wandb.Table(
            dataframe=pd.DataFrame(gat_graph_visualiser.node_data)
        )
        visualisations["scene_graph_edges"] = wandb.Table(
            dataframe=pd.DataFrame(gat_graph_visualiser.edge_data)
        )

        # Remove forward hooks
        for handle in handles:
            handle.remove()

        # Reset model to train
        self.model.train()

        return visualisations
