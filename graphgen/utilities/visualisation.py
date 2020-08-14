"""Tools for visualising images, bounding boxes and attention maps."""
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


def plot_image(
    image: np.array,
    output: Path,
    bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
) -> None:
    """Plot and save an image with optional bounding boxes to file."""
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    if bboxes is not None:
        for bbox in bboxes:
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            draw.rectangle(bbox, outline=color)
    img.save(output)


def plot_spatial_features(features: np.ndarray, output: Path) -> None:
    """Plot and save an spatial image features to file."""
    width = 64
    height = 32
    if features.shape[0] != width * height:
        raise ValueError(
            f"First dimension of param {features=} must be {width * height}"
        )
    hstep = features.shape[2]
    vstep = features.shape[1]
    image = np.zeros((height * vstep, width * hstep))
    for i in range(width):
        for j in range(height):
            image[j * vstep : (j + 1) * vstep, i * hstep : (i + 1) * hstep] = features[
                j * width + i
            ]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = (image * 255).astype(np.uint8)
    img = Image.fromarray(image)
    img.save(output)
