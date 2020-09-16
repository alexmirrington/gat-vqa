"""Classes storing model configuration information."""
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union


class ModelName(Enum):
    """Enum specifying possible model names."""

    FASTER_RCNN = "faster_rcnn"
    E2E_MULTI_GCN = "e2e_multi_gcn"
    MULTI_GCN = "multi_gcn"
    VQA = "vqa"
    GCN = "gcn"
    LSTM = "lstm"


class GCNConvName(Enum):
    """Enum specifying possible GCN conv layer names."""

    GCN = "gcn"
    GAT = "gat"


class GCNPoolingName(Enum):
    """Enum specifying possible GCN model names."""

    GLOBAL_MEAN = "global_mean"


class EmbeddingName(Enum):
    """Enum specifying possible embedding names."""

    GLOVE = "glove"
    ONE_HOT = "one_hot"
    STD_NORMAL = "std_normal"


class ReasoningModelName(Enum):
    """Enum specifying possible reasoning model names."""

    MAC = "mac"
    BOTTOM_UP = "bottom_up"


@dataclass
class ModelConfig:
    """Class for storing general model configuration information."""

    name: ModelName


@dataclass
class GCNModelConfig(ModelConfig):
    """Class for storing GCN model configuration information."""

    conv: GCNConvName
    shape: List[int]
    pooling: Optional[GCNPoolingName]
    heads: int

    def __post_init__(self) -> None:
        """Perform post-init checks on fields."""
        if self.name != ModelName.GCN:
            raise ValueError(f"Field {self.name=} must be equal to {ModelName.GCN}")
        if self.conv == GCNConvName.GAT and self.heads <= 0:
            raise ValueError(
                f"Field {self.heads=} must be positive for conv type {GCNConvName.GAT}"
            )
        if self.conv != GCNConvName.GAT and self.heads != 0:
            raise ValueError(
                f"Field {self.heads=} must be zero for non-attention GCNs."
            )


@dataclass
class LSTMModelConfig(ModelConfig):
    """Class for storing LSTM model configuration information."""

    input_dim: int
    hidden_dim: int
    bidirectional: bool

    def __post_init__(self) -> None:
        """Perform post-init checks on fields."""
        if self.name != ModelName.LSTM:
            raise ValueError(f"Field {self.name=} must be equal to {ModelName.LSTM}")


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
class EmbeddingConfig:
    """Class for storing embedding configuration information."""

    init: EmbeddingName
    dim: int
    trainable: bool


@dataclass
class EmbeddingModuleConfig:
    """Class for storing embedding module configuration information."""

    embedding: EmbeddingConfig
    module: Union[LSTMModelConfig, GCNModelConfig]


@dataclass
class VQAModelConfig(ModelConfig):
    """Class for storing model configuration information."""

    reasoning: Union[MACModelConfig, BottomUpModelConfig]
    question: EmbeddingModuleConfig
    scene_graph: EmbeddingModuleConfig

    def __post_init__(self) -> None:
        """Perform post-init checks on fields."""
        if self.name != ModelName.VQA:
            raise ValueError(f"Field {self.name=} must be equal to {ModelName.VQA}")


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


class Backbone(Enum):
    """Enum specifying possible backbones for the FasterRCNN object detector."""

    RESNET50 = "resnet50"


@dataclass
class BackboneConfig:
    """Class for storing R-CNN modlue configuration information."""

    name: Backbone
    pretrained: bool


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
