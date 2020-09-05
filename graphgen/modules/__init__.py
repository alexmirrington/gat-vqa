"""Package containing torch modules implementations."""
from .e2e_multi_gcn import E2EMultiGCN as E2EMultiGCN
from .faster_rcnn import FasterRCNN as FasterRCNN
from .gat import GAT as GAT
from .gcn import GCN as GCN
from .graph_rcnn import GraphRCNN as GraphRCNN
from .multi_gcn import MultiGCN as MultiGCN

__all__ = [
    MultiGCN.__name__,
    E2EMultiGCN.__name__,
    GraphRCNN.__name__,
    GCN.__name__,
    GAT.__name__,
    FasterRCNN.__name__,
]
