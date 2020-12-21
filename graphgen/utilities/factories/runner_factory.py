"""Utilities for creating models from a config."""

from typing import Optional, Sequence

import torch.nn
from torch_geometric.nn import global_mean_pool
from torchtext.vocab import GloVe

from ...config import Config
from ...config.model import (
    BottomUpModelConfig,
    EmbeddingConfig,
    EmbeddingName,
    GCNConvName,
    GCNModelConfig,
    GCNPoolingName,
    IdentityModelConfig,
    LSTMModelConfig,
    MACModelConfig,
    ModelName,
    SceneGraphAggregationName,
    TextCNNModelConfig,
    VQAModelConfig,
)
from ...config.training import OptimiserName
from ...modules import VQA
from ...modules.question import (
    CNNQuestionModule,
    GCNQuestionModule,
    IdentityQuestionModule,
    RNNQuestionModule,
)
from ...modules.reasoning.bottomup import BottomUp
from ...modules.reasoning.mac import MACCell, MACNetwork
from ...modules.reasoning.mac.control import ControlUnit
from ...modules.reasoning.mac.output import OutputUnit
from ...modules.reasoning.mac.read import ReadUnit
from ...modules.reasoning.mac.write import WriteUnit
from ...modules.scene import (
    GCNSceneGraphModule,
    IdentitySceneGraphModule,
    RNNSceneGraphModule,
)
from ...modules.sparse import GAT, GCN, AbstractGCN
from ..preprocessing import DatasetCollection, PreprocessorCollection
from ..runners import ResumeInfo, Runner, VQAModelRunner


class RunnerFactory:
    """Factory for creating runners from a config."""

    def __init__(self) -> None:
        """Create a RunnerFactory instance."""
        self._factory_methods = {
            ModelName.VQA: RunnerFactory.create_vqa,
        }

    @staticmethod
    def _build_optimiser(
        config: Config,
        model: torch.nn.Module,
    ) -> torch.optim.Optimizer:
        """Build an optimiser from a config for a given model."""
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
        elif config.training.optimiser.name == OptimiserName.ADADELTA:
            optimiser = torch.optim.Adadelta(
                model.parameters(),
                weight_decay=config.training.optimiser.weight_decay,
            )
        else:
            raise NotImplementedError()

        return optimiser

    @staticmethod
    def _build_gcn(config: GCNModelConfig, input_dim: int) -> AbstractGCN:
        """Build a GCN model from a config."""
        if config.pooling is None:
            pooling = None
        elif config.pooling == GCNPoolingName.GLOBAL_MEAN:
            pooling = global_mean_pool
        else:
            raise NotImplementedError()
        if config.conv == GCNConvName.GCN:
            return GCN(
                shape=[input_dim] + [config.dim for _ in range(config.layers)],
                pooling=pooling,
            )
        if config.conv == GCNConvName.GAT:
            return GAT(
                shape=[input_dim] + [config.dim for _ in range(config.layers)],
                pooling=pooling,
                heads=config.heads,
                concat=config.concat,
            )
        raise NotImplementedError()

    @staticmethod
    def _build_lstm(config: LSTMModelConfig, input_dim: int) -> torch.nn.LSTM:
        """Build a LSTM model from a config."""
        return torch.nn.LSTM(
            input_dim,
            config.hidden_dim // 2 if config.bidirectional else config.hidden_dim,
            batch_first=True,
            bidirectional=config.bidirectional,
        )

    @staticmethod
    def build_embeddings(  # TODO move to EmbeddingFactory
        config: EmbeddingConfig, words: Sequence[str]
    ) -> torch.nn.Embedding:
        """Build an embedding tensor."""
        if config.init == EmbeddingName.GLOVE:
            vectors = GloVe(name="6B", dim=config.dim)
            # Handle OOV
            if config.average_mwt:
                # Split word by space e.g. "next to" -> ["next", "to"]
                # so we can average tokens (esp. useful for relations)
                embeds = torch.stack(
                    [
                        torch.mean(
                            vectors.get_vecs_by_tokens(
                                word.split(), lower_case_backup=True
                            ),
                            dim=0,
                        )
                        for word in words
                    ],
                    dim=0,
                )
            else:
                embeds = vectors.get_vecs_by_tokens(words, lower_case_backup=True)
            return torch.nn.Embedding.from_pretrained(
                embeds, freeze=not config.trainable
            )
        if config.init == EmbeddingName.ONE_HOT:
            vectors = torch.eye(len(words))
            return torch.nn.Embedding.from_pretrained(
                vectors, freeze=not config.trainable
            )
        if config.init == EmbeddingName.STD_NORMAL:
            embedding = torch.nn.Embedding(
                num_embeddings=len(words),
                embedding_dim=config.dim,
            )
            embedding.weight.requires_grad = config.trainable
            return embedding
        raise NotImplementedError()

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
    def create_vqa(
        config: Config,
        device: torch.device,
        preprocessors: PreprocessorCollection,
        datasets: DatasetCollection,
        resume: Optional[ResumeInfo],
    ) -> Runner:
        """Create a runner from a config."""
        # pylint: disable=too-many-branches,too-many-locals,too-many-statements
        if not isinstance(config.model, VQAModelConfig):
            raise TypeError(
                f"Expected model config of type \
                {VQAModelConfig.__name__} \
                but got {config.model.__class__.__name__}"
            )

        num_answer_classes = len(preprocessors.questions.index_to_answer)

        # Create question module
        if isinstance(config.model.question.module, LSTMModelConfig):
            rnn = RunnerFactory._build_lstm(
                config.model.question.module, config.model.question.embedding.dim
            )
            question_module = RNNQuestionModule(rnn)
        elif isinstance(config.model.question.module, GCNModelConfig):
            gcn = RunnerFactory._build_gcn(
                config.model.question.module, config.model.question.embedding.dim
            )
            question_module = GCNQuestionModule(gcn)
        elif isinstance(config.model.question.module, TextCNNModelConfig):
            question_module = CNNQuestionModule(
                input_dim=config.model.question.embedding.dim,
                out_channels=config.model.reasoning.hidden_dim,
            )
        elif isinstance(config.model.question.module, IdentityModelConfig):
            question_module = IdentityQuestionModule()
        else:
            raise NotImplementedError()

        # Create scene graph module
        if isinstance(config.model.scene_graph.module, LSTMModelConfig):
            input_dim = config.model.scene_graph.embedding.dim
            if (
                config.model.scene_graph.graph.aggregation
                == SceneGraphAggregationName.PER_OBJ_CONCAT_MEAN_REL_ATTR
            ):
                input_dim = 3 * config.model.scene_graph.embedding.dim
            rnn = RunnerFactory._build_lstm(config.model.scene_graph.module, input_dim)
            scene_graph_module = RNNSceneGraphModule(rnn)
        elif isinstance(config.model.scene_graph.module, GCNModelConfig):
            gcn = RunnerFactory._build_gcn(
                config.model.scene_graph.module, config.model.scene_graph.embedding.dim
            )
            scene_graph_module = GCNSceneGraphModule(gcn)
        elif isinstance(config.model.scene_graph.module, IdentityModelConfig):
            scene_graph_module = IdentitySceneGraphModule()
        else:
            raise NotImplementedError()

        # Create reasoning module
        if isinstance(config.model.question.module, LSTMModelConfig):
            question_dim = config.model.question.module.hidden_dim
        elif isinstance(config.model.question.module, GCNModelConfig):
            question_dim = config.model.question.module.dim
        elif isinstance(config.model.question.module, TextCNNModelConfig):
            question_dim = config.model.reasoning.hidden_dim
        elif isinstance(config.model.question.module, IdentityModelConfig):
            question_dim = config.model.question.embedding.dim
        else:
            raise NotImplementedError()
        if isinstance(config.model.scene_graph.module, LSTMModelConfig):
            scene_graph_dim = config.model.scene_graph.module.hidden_dim
        elif isinstance(config.model.scene_graph.module, GCNModelConfig):
            scene_graph_dim = config.model.scene_graph.module.dim
        elif isinstance(config.model.scene_graph.module, IdentityModelConfig):
            scene_graph_dim = config.model.scene_graph.embedding.dim
        else:
            raise NotImplementedError()
        if isinstance(config.model.reasoning, MACModelConfig):
            reasoning_module = MACNetwork(
                length=config.model.reasoning.length,
                hidden_dim=config.model.reasoning.hidden_dim,
                question_dim=question_dim,
                knowledge_dim=scene_graph_dim,
                project_inputs=True,
                classifier=OutputUnit(
                    config.model.reasoning.hidden_dim, num_answer_classes
                ),
                cell=MACCell(
                    config.model.reasoning.hidden_dim,
                    control=ControlUnit(
                        control_dim=config.model.reasoning.hidden_dim,
                        length=config.model.reasoning.length,
                    ),
                    read=ReadUnit(
                        memory_dim=config.model.reasoning.hidden_dim,
                    ),
                    write=WriteUnit(hidden_dim=config.model.reasoning.hidden_dim),
                ),
            )
        elif isinstance(config.model.reasoning, BottomUpModelConfig):
            reasoning_module = BottomUp(
                question_dim=question_dim,
                knowledge_dim=scene_graph_dim,
                hidden_dim=config.model.reasoning.hidden_dim,
                output_dim=num_answer_classes,
            )

        # Create question embeddings
        question_embeddings = RunnerFactory.build_embeddings(
            config.model.question.embedding, preprocessors.questions.index_to_word
        )

        scene_graph_embeddings = None
        if config.model.scene_graph.embedding.trainable:
            # Create scene graph embeddings
            scene_graph_embeddings = RunnerFactory.build_embeddings(
                config.model.scene_graph.embedding,
                list(preprocessors.scene_graphs.object_to_index.keys())
                + list(preprocessors.scene_graphs.rel_to_index.keys())
                + list(preprocessors.scene_graphs.attr_to_index.keys()),
            )

        # Assemble model
        model = VQA(
            reasoning_module=reasoning_module,
            question_module=question_module,
            scene_graph_module=scene_graph_module,
            question_embeddings=question_embeddings,
            scene_graph_embeddings=scene_graph_embeddings,
        )
        optimiser = RunnerFactory._build_optimiser(config, model)
        criterion = torch.nn.NLLLoss()
        runner = VQAModelRunner(
            config, device, model, optimiser, criterion, datasets, preprocessors, resume
        )

        return runner
