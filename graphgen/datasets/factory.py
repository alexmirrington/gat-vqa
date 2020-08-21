"""Tools for creating datasets given configuration objects."""
from pathlib import Path
from typing import Dict, Optional

import torch.utils.data

from ..config import Config
from ..config.clevr import CLEVRDatasetConfig
from ..config.dataset import DatasetName
from ..config.gqa import GQADatasetConfig, GQAFeatures
from ..utilities.preprocessing import GQAQuestionPreprocessor, QuestionTransformer
from .gqa import GQA, GQAImages, GQAObjects, GQAQuestions, GQASceneGraphs, GQASpatial


class DatasetFactory:
    """Factory class for creating datasets given a configuration object."""

    def __init__(self) -> None:
        """Initialise the dataset factory."""
        self._factory_methods = {
            DatasetName.GQA: DatasetFactory._create_gqa,
            DatasetName.CLEVR: DatasetFactory._create_clevr,
        }

    def create(self, config: Config) -> torch.utils.data.Dataset:
        """Create a dataset from a given config."""
        return self._factory_methods[config.dataset.name](config)

    @staticmethod
    def _create_clevr(config: Config) -> torch.utils.data.Dataset:
        if not isinstance(config.dataset, CLEVRDatasetConfig):
            raise ValueError(
                f"Param {config.dataset=} must be of type",
                f"{CLEVRDatasetConfig.__name__}.",
            )
        raise NotImplementedError()

    @staticmethod
    def _create_gqa(config: Config) -> GQA:
        if not isinstance(config.dataset, GQADatasetConfig):
            raise ValueError(
                f"Param {config.dataset=} must be of type {GQADatasetConfig.__name__}."
            )
        dataset_config: GQADatasetConfig = config.dataset

        # Parse preprocessing pipeline caches
        caches: Dict[str, Optional[Path]] = {
            feat.value: None for feat in iter(GQAFeatures)
        }
        for item in config.preprocessing.pipeline:
            caches[item.feature] = (
                Path("cache")
                / dataset_config.name.value
                / item.feature
                / dataset_config.split.value
                / dataset_config.version.value
            )  # TODO don't hardcode "cache" path

        questions = GQAQuestions(
            dataset_config.filemap,
            dataset_config.split,
            dataset_config.version,
            cache=caches[GQAFeatures.QUESTIONS.value],
            preprocessor=GQAQuestionPreprocessor(),
            transform=QuestionTransformer(),
        )

        images = None
        objects = None
        spatial = None
        scene_graphs = None

        for feature in config.dataset.features:
            if feature == GQAFeatures.QUESTIONS:
                continue
            if feature == GQAFeatures.IMAGES:
                images = GQAImages(dataset_config.filemap)
            elif feature == GQAFeatures.OBJECTS:
                objects = GQAObjects(dataset_config.filemap)
            elif feature == GQAFeatures.SPATIAL:
                spatial = GQASpatial(dataset_config.filemap)
            elif feature == GQAFeatures.SCENE_GRAPHS:
                spatial = GQASceneGraphs(
                    dataset_config.filemap,
                    dataset_config.split,
                    cache=caches[GQAFeatures.SCENE_GRAPHS.value],
                )
            else:
                raise NotImplementedError()

        return GQA(
            questions,
            images=images,
            objects=objects,
            spatial=spatial,
            scene_graphs=scene_graphs,
        )
