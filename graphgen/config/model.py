"""Classes storing model configuration information."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union


class ModelName(Enum):
    """Enum specifying possible model names."""

    VQA = "vqa"
    GCN = "gcn"
    LSTM = "lstm"
    TEXT_CNN = "text_cnn"
    IDENTITY = "identity"


class SceneGraphAggregationName(Enum):
    """Enum specifying possible scene graph names."""

    PER_OBJ_CONCAT_MEAN_REL_ATTR = "per_obj_concat_mean_rel_attr"


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
    layers: int
    pooling: Optional[GCNPoolingName]
    heads: int
    concat: bool
    dim: int

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
        if self.layers <= 0:
            raise ValueError(f"Field {self.layers=} must be non-negative.")


@dataclass
class LSTMModelConfig(ModelConfig):
    """Class for storing LSTM model configuration information."""

    hidden_dim: int
    bidirectional: bool

    def __post_init__(self) -> None:
        """Perform post-init checks on fields."""
        if self.name != ModelName.LSTM:
            raise ValueError(f"Field {self.name=} must be equal to {ModelName.LSTM}")


@dataclass
class IdentityModelConfig(ModelConfig):
    """Class for storing identity module configuration information."""

    def __post_init__(self) -> None:
        """Perform post-init checks on fields."""
        if self.name != ModelName.IDENTITY:
            raise ValueError(
                f"Field {self.name=} must be equal to {ModelName.IDENTITY}"
            )


@dataclass
class TextCNNModelConfig(ModelConfig):
    """Class for storing text cnn model configuration information."""

    def __post_init__(self) -> None:
        """Perform post-init checks on fields."""
        if self.name != ModelName.TEXT_CNN:
            raise ValueError(
                f"Field {self.name=} must be equal to {ModelName.TEXT_CNN}"
            )


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
    average_mwt: bool
    dim: int
    trainable: bool


@dataclass
class SceneGraphConfig:
    """Class for storing scene graph configuration information."""

    directed: bool
    object_skip_edges: bool
    aggregation: Optional[SceneGraphAggregationName]

    def __post_init__(self) -> None:
        """Perform post-init checks on fields."""
        if not self.directed:
            raise ValueError(f"Field {self.directed=} must be True")


@dataclass
class EmbeddingModuleConfig:
    """Class for storing embedding module configuration information."""

    embedding: EmbeddingConfig
    module: Union[
        LSTMModelConfig, GCNModelConfig, TextCNNModelConfig, IdentityModelConfig
    ]


@dataclass
class SceneGraphEmbeddingModuleConfig(EmbeddingModuleConfig):
    """Class for storing scene graph embedding module configuration information."""

    graph: SceneGraphConfig


@dataclass
class VQAModelConfig(ModelConfig):
    """Class for storing model configuration information."""

    reasoning: Union[MACModelConfig, BottomUpModelConfig]
    question: EmbeddingModuleConfig
    scene_graph: SceneGraphEmbeddingModuleConfig

    def __post_init__(self) -> None:
        """Perform post-init checks on fields."""
        if self.name != ModelName.VQA:
            raise ValueError(f"Field {self.name=} must be equal to {ModelName.VQA}")
