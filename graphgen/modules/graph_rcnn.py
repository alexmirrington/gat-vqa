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

    def __init__(self, num_classes: int = 91, pretrained: bool = True) -> None:
        """Initialise the graph RCNN."""
        super().__init__()

        # TODO allow customisation of backbone etc.
        self.faster_rcnn = fasterrcnn_resnet50_fpn(
            pretrained=pretrained, num_classes=num_classes
        )
        self.repn = RePN(num_classes)

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[GraphRCNNTarget]] = None,
    ) -> Any:
        """Propagate data through the model."""
        if self.training and targets is None:
            raise ValueError("No targets given but model is in training mode.")

        # Register forward hooks
        # rpn_output_stack = []
        # bbox_head_output_stack = []
        # pooled_roi_feats_stack = []

        # self.faster_rcnn.roi_heads.box_predictor.register_forward_hook(
        #     lambda module, input, output: bbox_head_output_stack.append(
        #         output
        #     )  # TODO replace hook with forward method of the next part of the model
        # )
        # self.faster_rcnn.roi_heads.box_head.register_forward_hook(
        #     lambda module, input, output: pooled_roi_feats_stack.append(
        #         output
        #     )  # TODO replace hook with forward method of the next part of the model
        # )
        # self.faster_rcnn.rpn.register_forward_hook(
        #     lambda module, input, output: rpn_output_stack.append(
        #         output
        #     )
        # )

        # Perform RCNN forward pass
        rcnn_out = self.faster_rcnn(images, targets)

        # boxes, losses = rpn_output_stack.pop()  # Input to roi_head forward
        # class_logits, box_regression = bbox_head_output_stack.pop()
        # pooled_roi_feats = pooled_roi_feats_stack.pop()

        # Get relatedness between class logits. Entry i, j refers to relatedness
        # between subject i and object j
        # relatedness = self.repn(class_logits)

        # TODO filter i, i entries first?
        # TODO non-maximal suppression\
        # torch.sort(relatedness.view(-1), descending=True)
        # print(f"{pooled_roi_feats.size()=}")

        # print(f"{class_logits.size()=}")
        # print(f"{box_regression.size()=}")
        # print(f"{len(boxes)=}")
        # print(f"{boxes[0].size()}")

        # print(f"{torch.argmax(class_logits, dim=1)=}")
        # print(f"{box_regression=}")
        # print(f"{pooled_roi_feats=}")

        # print(f"{boxes=}")
        # print(f"{box_regression=}")

        return rcnn_out


class RePN(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method
    """Relation Proposal Network from "Graph R-CNN for Scene Graph Generation".

    References:
    -----------
    Graph R-CNN for Scene Graph Generation.
    Jianwei Yang, Jiasen Lu, Stefan Lee, Dhruv Batra and Devi Parikh.
    https://arxiv.org/pdf/1808.00191.pdf
    """

    def __init__(self, in_features: int) -> None:
        """Create a RePN module."""
        super().__init__()
        # TODO positional encoding?
        # https://github.com/alexmirrington/graph-rcnn.pytorch/blob/master/
        # lib/scene_parser/rcnn/modeling/relation_heads/relpn/relpn.py
        self.subject_projection = torch.nn.Sequential(  # phi in the paper
            torch.nn.Linear(in_features, 64),
            torch.nn.ReLU(True),
            torch.nn.Linear(64, 64),
        )
        self.object_projection = torch.nn.Sequential(  # psi in the paper
            torch.nn.Linear(in_features, 64),
            torch.nn.ReLU(True),
            torch.nn.Linear(64, 64),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Propagate data through the model."""
        subj_data = self.subject_projection(data)  # k x 64, k is number of classes
        obj_data = self.object_projection(data)  # k x 64
        scores = torch.mm(subj_data, obj_data.t())  # k x k
        return torch.sigmoid(scores)  # Sigmoid for 0-1 clamp
