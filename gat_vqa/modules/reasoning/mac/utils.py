"""A suite of utilities for MAC netowrk components."""
from torch import nn
from torch.nn.init import xavier_uniform_


def xavier_uniform_linear(in_dim: int, out_dim: int, bias: bool = True) -> nn.Linear:
    """Create a xavier-uniform-initialised linear layer."""
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()
    return lin
