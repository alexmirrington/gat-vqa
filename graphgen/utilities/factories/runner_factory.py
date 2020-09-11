"""Utilities for creating models from a config."""

from typing import Optional

import torch.nn
from torch_geometric.nn import global_mean_pool

from ...config import Config
from ...config.model import (
    Backbone,
    E2EMultiGCNModelConfig,
    FasterRCNNModelConfig,
    GCNModelConfig,
    GCNName,
    GCNPoolingName,
    LSTMModelConfig,
    MACMultiGCNModelConfig,
    ModelName,
    MultiGCNModelConfig,
)
from ...config.training import OptimiserName
from ...modules import (
    GAT,
    GCN,
    E2EMultiGCN,
    FasterRCNN,
    GraphRCNN,
    MACMultiGCN,
    MultiGCN,
)
from ...modules.mac import MACCell, MACNetwork
from ...modules.mac.control import ControlUnit
from ...modules.mac.output import OutputUnit
from ...modules.mac.read import ReadUnit
from ...modules.mac.write import WriteUnit
from ..preprocessing import DatasetCollection, PreprocessorCollection
from ..runners import (
    EndToEndMultiChannelGCNRunner,
    FasterRCNNRunner,
    MACMultiChannelGCNRunner,
    MultiChannelGCNRunner,
    ResumeInfo,
    Runner,
)


class RunnerFactory:
    """Factory for creating runners from a config."""

    def __init__(self) -> None:
        """Create a RunnerFactory instance."""
        self._factory_methods = {
            ModelName.FASTER_RCNN: RunnerFactory.create_faster_rcnn,
            ModelName.E2E_MULTI_GCN: RunnerFactory.create_e2emultigcn,
            ModelName.MULTI_GCN: RunnerFactory.create_multigcn,
            ModelName.MAC_MULTI_GCN: RunnerFactory.create_mac_multigcn,
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

    @staticmethod
    def create_mac_multigcn(
        config: Config,
        device: torch.device,
        preprocessors: PreprocessorCollection,
        datasets: DatasetCollection,
        resume: Optional[ResumeInfo],
    ) -> Runner:
        """Create an End-to-end multi-channel GCN runner from a config."""
        if not isinstance(config.model, MACMultiGCNModelConfig):
            raise TypeError(
                f"Expected model config of type {MACMultiGCNModelConfig.__name__} \
                but got {config.model.__class__.__name__}"
            )

        num_answer_classes = len(preprocessors.questions.index_to_answer)

        assert config.model.question is not None

        def build_gcn(config: GCNModelConfig) -> torch.nn.Module:
            if config.pooling is None:
                pool_func = None
            elif config.pooling == GCNPoolingName.GLOBAL_MEAN:
                pool_func = global_mean_pool
            else:
                raise NotImplementedError()
            if config.gcn == GCNName.GCN:
                return GCN(shape=config.layer_sizes, pool_func=pool_func)
            if config.gcn == GCNName.GAT:
                return GAT(shape=config.layer_sizes, heads=1, pool_func=pool_func)
            raise NotImplementedError()

        # Create question module
        if isinstance(config.model.question, LSTMModelConfig):
            question_module = torch.nn.LSTM(
                config.model.question.input_dim,
                config.model.question.hidden_dim // 2
                if config.model.question.bidirectional
                else config.model.question.hidden_dim,
                batch_first=True,
                bidirectional=config.model.question.bidirectional,
            )
        elif isinstance(config.model.question, GCNModelConfig):
            question_module = build_gcn(config.model.question)

        # Create scene gcn
        scene_gcn = None
        if config.model.scene_graph is not None:
            scene_gcn = build_gcn(config.model.scene_graph)

        model = MACMultiGCN(
            mac_network=MACNetwork(
                length=config.model.mac.length,
                hidden_dim=config.model.mac.hidden_dim,
                classifier=OutputUnit(config.model.mac.hidden_dim, num_answer_classes),
                cell=MACCell(
                    config.model.mac.hidden_dim,
                    control=ControlUnit(
                        control_dim=config.model.mac.hidden_dim,
                        length=config.model.mac.length,  # TODO set question input dim
                    ),
                    read=ReadUnit(
                        memory_dim=config.model.mac.hidden_dim,
                    ),
                    write=WriteUnit(hidden_dim=config.model.mac.hidden_dim),
                ),
            ),
            question_module=question_module,
            scene_gcn=scene_gcn,
        )
        optimiser = RunnerFactory._build_optimiser(config, model)
        criterion = torch.nn.NLLLoss()
        runner = MACMultiChannelGCNRunner(
            config, device, model, optimiser, criterion, datasets, preprocessors, resume
        )

        return runner
