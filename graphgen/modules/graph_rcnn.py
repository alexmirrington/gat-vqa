"""Module containing code for a FasterRCNN object detector."""
from typing import Any, List, Optional, Tuple

import torch
from torch_geometric.data import Batch, Data
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from .faster_rcnn import FasterRCNNTarget
from .relpn import RelPN


class GraphRCNN(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method
    """A GraphRCNN model."""

    def __init__(self, num_classes: int = 91, pretrained: bool = True) -> None:
        """Initialise the graph RCNN."""
        super().__init__()

        # Create submodules
        self.faster_rcnn = fasterrcnn_resnet50_fpn(  # TODO allow customisation
            pretrained=pretrained, num_classes=num_classes
        )
        self.relpn = RelPN(num_classes)

        # Register forward hooks
        self.__bbox_head_outputs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.faster_rcnn.roi_heads.box_predictor.register_forward_hook(
            lambda module, input, output: self._rcnn_bbox_head_hook(*output)
        )

        self.__bbox_roi_pool_inputs: List[Any] = []
        self.faster_rcnn.roi_heads.box_roi_pool.register_forward_hook(
            self._rcnn_box_roi_pool_hook
        )

    def _rcnn_bbox_head_hook(
        self, class_logits: torch.Tensor, box_regression: torch.Tensor
    ) -> None:
        self.__bbox_head_outputs.append((class_logits, box_regression))

    def _rcnn_box_roi_pool_hook(
        self, module: Any, inputs: Any, outputs: Any  # pylint: disable=unused-argument
    ) -> None:
        # features, proposals, image_shapes = inputs
        # features is collections.OrderedDict, feature maps from FPN layers.
        # proposals list of tensors of shape [512, 4] each.
        # image_shapes is a list of tuples

        # box_features = outputs
        # box_features has shape [num_proposals * num_images, 256, 7, 7], output of FPN
        # Last 3 dims are flattened and passed to box_predictor
        self.__bbox_roi_pool_inputs.append(inputs)

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[FasterRCNNTarget]] = None,
    ) -> Any:
        """Propagate data through the model."""
        if self.training and targets is None:
            raise ValueError("No targets given but model is in training mode.")

        # TODO Capture sub-targets in targets dict? RCNN will work on COCO
        # classes but we can use sub-targets to help fine-tune?

        # Perform RCNN forward pass, triggering forward hooks
        rcnn_out = self.faster_rcnn(images, targets)

        # Retrieve tensors
        class_logits, box_regression = self.__bbox_head_outputs.pop()
        (
            features,
            proposals,
            image_shapes,
        ) = self.__bbox_roi_pool_inputs.pop()  # features, proposals, image_shapes

        # Ensure empty stacks to avoid running out of memory and prevent
        # recursive behaviour
        assert len(self.__bbox_head_outputs) == 0
        assert len(self.__bbox_roi_pool_inputs) == 0

        # class_logits is aTensor of shape (num_proposals * num_images, num_classes),
        # we need to split it up to perform per-image processing in RelPN
        per_image_class_logits = torch.chunk(class_logits, chunks=len(images), dim=0)
        # assert torch.all(torch.cat(per_image_class_logits, dim=0) == class_logits)

        # Get RelPN relationship predictions
        relations, scores = self.relpn(per_image_class_logits, proposals)

        # Use img_logits directly, as img_relations are indexed relative to
        # number of proposals
        semantic_adjacency = Batch.from_data_list(
            [
                Data(
                    edge_index=img_relations.t(),
                    edge_attr=img_scores.unsqueeze(-1),
                    x=img_logits,
                )
                for img_relations, img_scores, img_logits in zip(
                    relations, scores, per_image_class_logits
                )
            ]
        )  # x has shape [num_nodes, num_node_features]
        return rcnn_out, semantic_adjacency
