"""Package containing torch modules implementations."""
from .gcn import GCN as GCN
from .graph_rcnn import GraphRCNN as GraphRCNN
from .multichannel_gcn import MultiGCN as MultiGCN

__all__ = [MultiGCN.__name__, GraphRCNN.__name__, GCN.__name__]
