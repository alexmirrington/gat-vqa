"""Module containing code for a FasterRCNN object detector."""
from typing import Any, List, Optional, TypedDict

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class FasterRCNNTarget(TypedDict):
    """TypedDict specifying required fields for training a faster RCNN model."""

    boxes: torch.FloatTensor
    labels: torch.IntTensor


class FasterRCNN(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method
    """Wrapper for a FasterRCNN model, mainly for typing purposes."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        pretrained_backbone: bool = True,
    ):
        """Initialise the faster RCNN model."""
        super().__init__()

        # Create submodules
        self.faster_rcnn = fasterrcnn_resnet50_fpn(
            pretrained=pretrained,
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
        )

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[FasterRCNNTarget]] = None,
    ) -> Any:
        """Propagate data through the model."""
        return self.faster_rcnn(images, targets)
