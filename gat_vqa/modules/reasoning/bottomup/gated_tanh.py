"""Implementation of a gated tanh layer module.

References:
-----------
https://openaccess.thecvf.com/content_cvpr_2018/html/Anderson_Bottom-Up_and_\
Top-Down_CVPR_2018_paper.html
"""
import torch


class GatedTanh(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method  # noqa: B950
    """A gated tanh layer module."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Initialise a gated tanh layer."""
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.tanh_layer = torch.nn.Linear(input_dim, output_dim, bias=True)
        self.gate_layer = torch.nn.Linear(input_dim, output_dim, bias=True)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the module's parameters."""
        # torch.nn.init.xavier_uniform_(
        #     self.tanh_layer.weight, gain=torch.nn.init.calculate_gain("tanh")
        # )
        # torch.nn.init.xavier_uniform_(
        #     self.gate_layer.weight, gain=torch.nn.init.calculate_gain("sigmoid")
        # )
        torch.nn.init.xavier_uniform_(
            self.gate_layer.weight, gain=torch.nn.init.calculate_gain("relu")
        )

    def forward(self, data: torch.tensor) -> torch.Tensor:
        """Propagate data through the model.

        Refer to `torch.nn.Linear` for compatible input and output shapes.
        """
        # out = torch.tanh(self.tanh_layer(data))
        # gate = torch.sigmoid(self.gate_layer(data))
        # return out * gate
        return torch.relu(self.gate_layer(data))
