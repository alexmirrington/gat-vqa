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

    def __init__(self, num_classes: int) -> None:
        """Initialise the graph RCNN."""
        super().__init__()

        # TODO allow customisation of backbone etc.
        self.faster_rcnn = fasterrcnn_resnet50_fpn(num_classes=num_classes)

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[GraphRCNNTarget]] = None,
    ) -> Any:
        """Propagate data through the model."""
        if self.training and targets is None:
            raise ValueError("No targets given but model is in training mode.")

        bbox_intermediate = []
        self.faster_rcnn.roi_heads.box_predictor.register_forward_hook(
            lambda module, input, output: bbox_intermediate.append(
                output
            )  # TODO replace hook with forward method of the next part of the model
        )
        rcnn_out = self.faster_rcnn(images, targets)
        # Capture intermediate outputs of ROI head bboxes
        # [
        #   losses: torch.Size([num_rois, num_classes]),
        #   boxes: torch.Size([num_rois, 4*num_classes])
        # ]
        bbox_losses, bbox_preds = bbox_intermediate.pop()
        print(f"{bbox_losses=}")
        print(f"{bbox_preds=}")

        # TODO retrieve intermediate pooled features from after ROI pooling
        # layer but before class predictor and bbox regressor via hook

        # TODO retrieve class predictor loss and preds via hook

        return rcnn_out
