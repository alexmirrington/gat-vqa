"""Module containing tools for visualising attention maps and graphs."""
from typing import Dict, Optional, Sequence

import wandb
from torch import Tensor

from ..schemas.common import BoundingBox


def wandb_image(
    image: Tensor,
    caption: Optional[str] = None,
    boxes: Optional[Dict[str, Sequence[BoundingBox]]] = None,
    object_to_index: Optional[Dict[str, int]] = None,
) -> wandb.Image:
    """Create a list of wandb images for logging iwth optional bounding boxes."""
    class_labels = (
        {idx: key for key, idx in object_to_index.items()}
        if object_to_index is not None
        else None
    )
    wandb_boxes = (
        {
            key: {
                "box_data": [
                    {
                        "position": {
                            "minX": box.min_x,
                            "maxX": box.max_x,
                            "minY": box.min_y,
                            "maxY": box.max_y,
                        },
                        "class_id": box.label,
                        "box_caption": class_labels[box.label]
                        if class_labels is not None and box.label is not None
                        else box.label,
                        "domain": "pixel",
                    }
                    for box in keyed_boxes
                ],
                "class_labels": class_labels,
            }
            for key, keyed_boxes in boxes.items()
        }
        if boxes is not None
        else None
    )
    return wandb.Image(image, boxes=wandb_boxes, caption=caption)
