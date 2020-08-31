"""Utilities for creating models from a config."""

from typing import Optional

import torch.nn

from ...config import Config
from ...config.model import (
    Backbone,
    FasterRCNNModelConfig,
    ModelName,
    MultiGCNModelConfig,
)
from ...config.training import OptimiserName
from ...modules import GCN, FasterRCNN, GraphRCNN, MultiGCN
from ...utilities.runners import (
    EndToEndMultiChannelGCNRunner,
    FasterRCNNRunner,
    ResumeInfo,
    Runner,
)
from .dataset_factory import DatasetCollection
from .preprocessing_factory import PreprocessorCollection


class RunnerFactory:
    """Factory for creating runners from a config."""

    def __init__(self) -> None:
        """Create a RunnerFactory instance."""
        self._factory_methods = {
            ModelName.FASTER_RCNN: RunnerFactory.create_faster_rcnn,
            ModelName.MULTICHANNEL_GCN: RunnerFactory.create_multichannel_gcn,
        }

    @staticmethod
    def _build_optimiser(
        config: Config,
        model: torch.nn.Module,
    ) -> torch.optim.Optimizer:
        if config.training.optimiser.name == OptimiserName.ADAM:
            optimiser = torch.optim.Adam(
                model.parameters(),
                lr=config.training.optimiser.learning_rate,
                weight_decay=config.training.optimiser.weight_decay,
            )
        elif config.training.optimiser.name == OptimiserName.SGD:
            optimiser = torch.optim.SGD(
                model.parameters(),
                lr=config.training.optimiser.learning_rate,
                momentum=config.training.optimiser.momentum,
                weight_decay=config.training.optimiser.weight_decay,
            )
        else:
            raise NotImplementedError()

        return optimiser

    def create(
        self,
        config: Config,
        device: torch.device,
        preprocessors: PreprocessorCollection,
        datasets: DatasetCollection,
        resume: Optional[ResumeInfo],
    ) -> Runner:
        """Create a runner from a config."""
        return self._factory_methods[config.model.name](
            config, device, preprocessors, datasets, resume
        )

    @staticmethod
    def create_faster_rcnn(
        config: Config,
        device: torch.device,
        preprocessors: PreprocessorCollection,
        datasets: DatasetCollection,
        resume: Optional[ResumeInfo],
    ) -> Runner:
        """Create a Faster-RCNN runner from a config."""
        if not isinstance(config.model, FasterRCNNModelConfig):
            raise TypeError(
                f"Expected model config of type {FasterRCNNModelConfig.__name__} \
                but got {config.model.__class__.__name__}"
            )

        # Add one for background
        num_classes = len(set(preprocessors.scene_graphs.object_to_index.values()))

        if config.model.backbone.name == Backbone.RESNET50:
            model = FasterRCNN(
                pretrained=config.model.pretrained,
                num_classes=num_classes,
                pretrained_backbone=config.model.backbone.pretrained,
            )
        else:
            raise NotImplementedError()

        optimiser = RunnerFactory._build_optimiser(config, model)
        runner = FasterRCNNRunner(
            config, device, model, optimiser, None, datasets, preprocessors, resume
        )

        return runner

    @staticmethod
    def create_multichannel_gcn(
        config: Config,
        device: torch.device,
        preprocessors: PreprocessorCollection,
        datasets: DatasetCollection,
        resume: Optional[ResumeInfo],
    ) -> Runner:
        """Create an End-to-end multi-channel GCN runner from a config."""
        if not isinstance(config.model, MultiGCNModelConfig):
            raise TypeError(
                f"Expected model config of type {MultiGCNModelConfig.__name__} \
                but got {config.model.__class__.__name__}"
            )

        model = MultiGCN(
            len(preprocessors.questions.index_to_answer),
            GraphRCNN(num_classes=91, pretrained=True),
            dep_gcn=GCN((300, 512, 768, 1024)),
            obj_semantic_gcn=GCN((91, 256, 512, 1024)),
        )
        optimiser = RunnerFactory._build_optimiser(config, model)
        criterion = torch.nn.NLLLoss()
        runner = EndToEndMultiChannelGCNRunner(
            config, device, model, optimiser, criterion, datasets, preprocessors, resume
        )

        return runner
