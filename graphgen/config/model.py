"""Classes storing model configuration information."""
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union


class ModelName(Enum):
    """Enum specifying possible model names."""

    FASTER_RCNN = "faster_rcnn"
    E2E_MULTI_GCN = "e2e_multi_gcn"
    MULTI_GCN = "multi_gcn"
    REASONING_MULTI_GCN = "reasoning_multi_gcn"


class GCNName(Enum):
    """Enum specifying possible GCN model names."""

    GCN = "gcn"
    GAT = "gat"


class GCNPoolingName(Enum):
    """Enum specifying possible GCN model names."""

    GLOBAL_MEAN = "global_mean"


class EmbeddingName(Enum):
    """Enum specifying possible embedding names."""

    GLOVE = "glove"
    ONE_HOT = "one_hot"
    NORMAL = "normal"


class ReasoningModelName(Enum):
    """Enum specifying possible reasoning model names."""

    MAC = "mac"
    BOTTOM_UP = "bottom_up"


class Backbone(Enum):
    """Enum specifying possible backbones for the FasterRCNN object detector."""

    RESNET50 = "resnet50"


@dataclass
class BackboneConfig:
    """Class for storing R-CNN modlue configuration information."""

    name: Backbone
    pretrained: bool


@dataclass
class ModelConfig:
    """Class for storing general model configuration information."""

    name: ModelName


@dataclass
class FasterRCNNModelConfig(ModelConfig):
    """Class for storing model configuration information."""

    pretrained: bool
    backbone: BackboneConfig

    def __post_init__(self) -> None:
        """Perform post-init checks on fields."""
        if self.name != ModelName.FASTER_RCNN:
            raise ValueError(
                f"Field {self.name=} must be equal to {ModelName.FASTER_RCNN}."
            )


@dataclass
class E2EMultiGCNModelConfig(ModelConfig):
    """Class for storing model configuration information."""

    def __post_init__(self) -> None:
        """Perform post-init checks on fields."""
        if self.name != ModelName.E2E_MULTI_GCN:
            raise ValueError(
                f"Field {self.name=} must be equal to {ModelName.E2E_MULTI_GCN}"
            )


@dataclass
class GCNModelConfig:
    """Class for storing GCN model configuration information."""

    embedding_dim: int
    embedding: EmbeddingName
    gcn: GCNName
    pooling: Optional[GCNPoolingName]
    layer_sizes: List[int]


@dataclass
class MultiGCNModelConfig(ModelConfig):
    """Class for storing model configuration information."""

    text_syntactic_graph: Optional[GCNModelConfig]
    scene_graph: Optional[GCNModelConfig]

    def __post_init__(self) -> None:
        """Perform post-init checks on fields."""
        if self.name != ModelName.MULTI_GCN:
            raise ValueError(
                f"Field {self.name=} must be equal to {ModelName.MULTI_GCN}"
            )


@dataclass
class LSTMModelConfig:
    """Class for storing LSTM model configuration information."""

    embedding_dim: int
    embedding: EmbeddingName
    hidden_dim: int
    bidirectional: bool


@dataclass
class ReasoningModelConfig:
    """Class for storing general reasoning model configuration information."""

    name: ReasoningModelName


@dataclass
class MACModelConfig(ReasoningModelConfig):
    """Class for storing MAC network model configuration information."""

    length: int
    hidden_dim: int


@dataclass
class BottomUpModelConfig(ReasoningModelConfig):
    """Class for storing bottom-up model configuration information."""

    hidden_dim: int


@dataclass
class ReasoningMultiGCNModelConfig(ModelConfig):
    """Class for storing model configuration information."""

    reasoning: Union[MACModelConfig, BottomUpModelConfig]
    question: Union[LSTMModelConfig, GCNModelConfig]
    scene_graph: Union[LSTMModelConfig, GCNModelConfig]

    def __post_init__(self) -> None:
        """Perform post-init checks on fields."""
        if self.name != ModelName.REASONING_MULTI_GCN:
            raise ValueError(
                f"Field {self.name=} must be equal to {ModelName.REASONING_MULTI_GCN}"
            )
