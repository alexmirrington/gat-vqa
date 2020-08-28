"""Module containing code for a FasterRCNN object detector."""
from typing import Any, List, Optional, TypedDict

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class GraphRCNNTarget(TypedDict):
    """TypedDict specifying required fields for training a faster RCNN model."""

    boxes: torch.FloatTensor
    labels: torch.IntTensor


# REFER TO: https://github.com/pytorch/vision/issues/2500
# This will enable e2e training :D


# TODO consider whether to inherit from GeneralisedRCNN
class GraphRCNN(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method
    """A GraphRCNN model."""

    def __init__(self) -> None:
        """Initialise the graph RCNN."""
        super().__init__()

        # TODO allow customisation of backbone etc.
        self.faster_rcnn = fasterrcnn_resnet50_fpn()

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[GraphRCNNTarget]] = None,
    ) -> Any:
        """Propagate data through the model."""
        if self.training and targets is None:
            raise ValueError("No targets given but model is in training mode.")

        return self.faster_rcnn(images, targets)
