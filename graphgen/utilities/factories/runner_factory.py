"""Utilities for creating models from a config."""

from typing import Optional

import torch.nn
from torch_geometric.nn import global_mean_pool

from ...config import Config
from ...config.model import (
    Backbone,
    E2EMultiGCNModelConfig,
    FasterRCNNModelConfig,
    ModelName,
    MultiGCNModelConfig,
)
from ...config.training import OptimiserName
from ...modules import GCN, E2EMultiGCN, FasterRCNN, GraphRCNN, MultiGCN
from ...utilities.runners import (
    EndToEndMultiChannelGCNRunner,
    FasterRCNNRunner,
    MultiChannelGCNRunner,
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
            ModelName.E2E_MULTI_GCN: RunnerFactory.create_e2emultigcn,
            ModelName.MULTI_GCN: RunnerFactory.create_multigcn,
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
        print(num_classes)

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
    def create_e2emultigcn(
        config: Config,
        device: torch.device,
        preprocessors: PreprocessorCollection,
        datasets: DatasetCollection,
        resume: Optional[ResumeInfo],
    ) -> Runner:
        """Create an End-to-end multi-channel GCN runner from a config."""
        if not isinstance(config.model, E2EMultiGCNModelConfig):
            raise TypeError(
                f"Expected model config of type {E2EMultiGCNModelConfig.__name__} \
                but got {config.model.__class__.__name__}"
            )

        model = E2EMultiGCN(
            len(preprocessors.questions.index_to_answer),
            GraphRCNN(num_classes=91, pretrained=True),
            dep_gcn=GCN((300, 512, 768, 1024), global_mean_pool),
            obj_semantic_gcn=GCN((91, 256, 512, 1024), global_mean_pool),
        )
        optimiser = RunnerFactory._build_optimiser(config, model)
        criterion = torch.nn.NLLLoss()
        runner = EndToEndMultiChannelGCNRunner(
            config, device, model, optimiser, criterion, datasets, preprocessors, resume
        )

        return runner

    @staticmethod
    def create_multigcn(
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

        num_answer_classes = len(preprocessors.questions.index_to_answer)

        assert config.model.text_syntactic_graph is not None
        assert config.model.scene_graph is not None

        # Create GCN
        model = MultiGCN(
            num_answer_classes=num_answer_classes,
            text_gcn_shape=config.model.text_syntactic_graph.layer_sizes,
            text_gcn_conv=config.model.text_syntactic_graph.gcn,
            scene_gcn_shape=config.model.scene_graph.layer_sizes,
            scene_gcn_conv=config.model.scene_graph.gcn,
        )
        optimiser = RunnerFactory._build_optimiser(config, model)
        criterion = torch.nn.NLLLoss()
        runner = MultiChannelGCNRunner(
            config, device, model, optimiser, criterion, datasets, preprocessors, resume
        )

        return runner
