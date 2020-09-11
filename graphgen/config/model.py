"""Classes storing model configuration information."""
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union


class ModelName(Enum):
    """Enum specifying possible model names."""

    FASTER_RCNN = "faster_rcnn"
    E2E_MULTI_GCN = "e2e_multi_gcn"
    MULTI_GCN = "multi_gcn"
    MAC_MULTI_GCN = "mac_multi_gcn"


class GCNName(Enum):
    """Enum specifying possible GCN model names."""

    GCN = "gcn"
    GAT = "gat"


class GCNPoolingName(Enum):
    """Enum specifying possible GCN model names."""

    GLOBAL_MEAN = "global_mean"


class Backbone(Enum):
    """Enum specifying possible backbones for the FasterRCNN object detector."""

    RESNET50 = "resnet50"


@dataclass(frozen=True)
class BackboneConfig:
    """Class for storing R-CNN modlue configuration information."""

    name: Backbone
    pretrained: bool


@dataclass(frozen=True)
class ModelConfig:
    """Class for storing general model configuration information."""

    name: ModelName


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class E2EMultiGCNModelConfig(ModelConfig):
    """Class for storing model configuration information."""

    def __post_init__(self) -> None:
        """Perform post-init checks on fields."""
        if self.name != ModelName.E2E_MULTI_GCN:
            raise ValueError(
                f"Field {self.name=} must be equal to {ModelName.E2E_MULTI_GCN}"
            )


@dataclass(frozen=True)
class GCNModelConfig:
    """Class for storing GCN model configuration information."""

    gcn: GCNName
    pooling: Optional[GCNPoolingName]
    layer_sizes: List[int]


@dataclass(frozen=True)
class MACModelConfig:
    """Class for storing MAC network model configuration information."""

    length: int
    hidden_dim: int


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class LSTMModelConfig:
    """Class for storing LSTM model configuration information."""

    input_dim: int
    hidden_dim: int
    bidirectional: bool


@dataclass(frozen=True)
class MACMultiGCNModelConfig(ModelConfig):
    """Class for storing model configuration information."""

    mac: MACModelConfig
    question: Union[LSTMModelConfig, GCNModelConfig]
    scene_graph: Optional[GCNModelConfig]

    def __post_init__(self) -> None:
        """Perform post-init checks on fields."""
        if self.name != ModelName.MAC_MULTI_GCN:
            raise ValueError(
                f"Field {self.name=} must be equal to {ModelName.MAC_MULTI_GCN}"
            )
