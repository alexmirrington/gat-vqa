"""Package containing a MAC network implementation.

References:
-----------
Original paper: https://arxiv.org/abs/1803.03067
Original implementation: https://github.com/stanfordnlp/mac-network
PyTorch implementation: https://github.com/ceyzaguirre4/mac-network-pytorch
"""
from .network import MACCell as MACCell
from .network import MACNetwork as MACNetwork
from .network import OriginalMACNetwork as OriginalMACNetwork

__all__ = [OriginalMACNetwork.__name__, MACNetwork.__name__, MACCell.__name__]
