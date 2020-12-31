"""Tools for visualising images, bounding boxes and attention maps."""
import random
from typing import Dict, List, Optional, Sequence

import numpy as np
import torchvision.transforms as T
import wandb
from PIL import ImageDraw
from torch import Tensor

from ..schemas.common import BoundingBox


def plot_image(
    image: Tensor,
    caption: Optional[str] = None,
    boxes: Optional[Dict[str, Sequence[BoundingBox]]] = None,
) -> None:
    """Plot and save an image with optional bounding boxes to file."""
    tfm = T.ToPILImage()
    img = tfm(image)
    draw = ImageDraw.Draw(img)
    if boxes is not None:
        for bbox_seq in boxes.values():
            for bbox in bbox_seq:
                color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
                draw.rectangle(
                    (bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y),
                    outline=color,
                    width=2,
                )
    return wandb.Image(img, caption=caption)


def wandb_image(
    image: Tensor,
    caption: Optional[str] = None,
    boxes: Optional[Dict[str, Sequence[BoundingBox]]] = None,
    object_to_index: Optional[Dict[str, int]] = None,
) -> wandb.Image:
    """Create a list of wandb images for logging with optional bounding boxes."""
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
                            "minX": box.min_x / image.size(2),
                            "maxX": box.max_x / image.size(2),
                            "minY": box.min_y / image.size(1),
                            "maxY": box.max_y / image.size(1),
                        },
                        "class_id": box.label,
                        "box_caption": class_labels[box.label]
                        if class_labels is not None and box.label is not None
                        else box.label,
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


class SparseGraphVisualiser:

    MAX_ROWS: int = wandb.Table.MAX_ROWS

    def __init__(self):
        self.node_data: Dict[str, List] = {
            "sample": [],
            "step": [],
            "node_index": [],
            "node_label": [],
            "node_group": [],
        }
        self.edge_data: Dict[str, List] = {
            "sample": [],
            "head": [],
            "source_node_index": [],
            "target_node_index": [],
        }
        self.__index = 0

    def add_graph(
        self,
        node_labels: Sequence[str],
        node_groups: Sequence[str],
        edge_indices: Tensor,
        edge_values: Dict[str, Tensor] = {},
        node_values: Dict[str, Tensor] = {},
    ):
        # Convert tensors to numpy arrays
        edge_indices = edge_indices.detach().squeeze().cpu().numpy()
        edge_values = {
            key: val.detach().squeeze().cpu().numpy()
            for key, val in edge_values.items()
        }
        node_values = {
            key: val.detach().squeeze().cpu().numpy()
            for key, val in node_values.items()
        }
        # Perform length and size checks
        assert len(edge_indices.shape) == 2
        assert edge_indices.shape[0] == 2
        head_count = 1
        for t in edge_values.values():
            # Single head, expand dimensions so we can iterate over heads later
            if len(t.shape) == 1:
                np.expand_dims(t, axis=-1)
            assert len(t.shape) == 2
            head_count = t.shape[1]
            assert edge_indices.shape[1] == t.shape[0]

        assert len(node_labels) == len(node_groups)
        step_count = 1
        for t in node_values.values():
            if len(t.shape) == 1:
                np.expand_dims(t, axis=-1)
            assert len(t.shape) == 2
            step_count = t.shape[1]
            assert len(node_labels) == t.shape[0]

        # Assert adding the graph will not go over the row limit
        if (
            len(self.edge_data["sample"]) + head_count * edge_indices.shape[1]
            > self.MAX_ROWS
            or len(self.node_data["sample"]) + len(node_labels) > self.MAX_ROWS
        ):
            raise ValueError(
                f"Did not add graph to avoid exceeding row count of {self.MAX_ROWS}"
            )

        # Populate edge values
        for key, values in edge_values.items():
            # Offset missing samples if needed
            if key not in self.edge_data.keys():
                self.edge_data[key] = [None] * len(self.edge_data["sample"])
            for i in range(head_count):
                self.edge_data[key] += list(values[:, i])
        for head_idx in range(head_count):
            self.edge_data["head"] += [head_idx] * edge_indices.shape[1]
            self.edge_data["source_node_index"] += list(edge_indices[0, :])
            self.edge_data["target_node_index"] += list(edge_indices[1, :])
            self.edge_data["sample"] += [self.__index] * edge_indices.shape[1]

        # Check for any keys that weren't updated
        for key in self.edge_data.keys():
            if key not in (
                "source_node_index",
                "target_node_index",
                "sample",
                *edge_values.keys(),
            ):
                self.edge_data[key] += [None] * (
                    len(self.edge_data["sample"]) - len(self.edge_data[key])
                )

        # Populate node values
        for key, values in node_values.items():
            # Offset missing samples if needed
            if key not in self.node_data.keys():
                self.node_data[key] = [None] * len(self.node_data["sample"])
            for i in range(step_count):
                self.node_data[key] += list(values[:, i])
        for step_idx in range(step_count):
            self.node_data["step"] += [step_idx] * len(node_labels)
            self.node_data["node_index"] += list(range(len(node_labels)))
            self.node_data["node_label"] += node_labels
            self.node_data["node_group"] += node_groups
            self.node_data["sample"] += [self.__index] * len(node_labels)

        # Check for any keys that weren't updated
        for key in self.node_data.keys():
            if key not in (
                "sample",
                "node_index",
                "node_label",
                "node_group",
                *node_values.keys(),
            ):
                self.node_data[key] += [None] * (
                    len(self.node_data["sample"]) - len(self.node_data[key])
                )

        self.__index += 1


class AttentionMapVisualiser:

    def __init__(self):
        self.heatmap_data: Dict[str, List] = {
            "sample": [],
            "values": [],
            "x": [],
            "y": [],
            "x_label": [],
            "y_label": [],
        }
        self.__index = 0

    def add_attention_map(
        self,
        matrix: Tensor,
        x_labels: List[str],
        y_labels: List[str]
    ):
        matrix = matrix.detach().cpu().numpy()
        y, x = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
        x = list(x.T.flatten())
        y = list(y.T.flatten())
        values = list(matrix.flatten())
        # Assert matrix reshapes match up
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                assert (
                    abs(
                        float(values[row * matrix.shape[1] + col]) - float(matrix[row][col])
                    )
                    < 1e-4
                )
                assert col == int(x[row * matrix.shape[1] + col])
                assert row == int(y[row * matrix.shape[1] + col])
        table = wandb.Table(
            data=[[x_, y_, x_labels[x_], y_labels[y_], v] for x_, y_, v in zip(x, y, values)],
            columns=["x", "y", "x_label", "y_label", "value"],
        )

        self.heatmap_data["sample"] += [self.__index] * len(values)
        self.heatmap_data["x"] += x
        self.heatmap_data["y"] += y
        self.heatmap_data["values"] += values
        self.heatmap_data["x_label"] += [x_labels[x_] for x_ in x]
        self.heatmap_data["y_label"] += [y_labels[y_] for y_ in y]

        self.__index += 1