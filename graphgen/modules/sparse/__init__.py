"""Package containing modules for processing sparse graphs."""
from .abstract_gcn import AbstractGCN as AbstractGCN
from .gat import GAT as GAT
from .gcn import GCN as GCN

__all__ = [AbstractGCN.__name__, GCN.__name__, GAT.__name__]
