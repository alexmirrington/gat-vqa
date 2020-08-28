"""Tools for creating trainable datasets given configuration objects."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import jsons
import wandb
from torch.utils.data import Dataset

from ...config import Config
from ...config.clevr import CLEVRDatasetConfig
from ...config.dataset import DatasetName
from ...config.gqa import (
    GQADatasetConfig,
    GQAFeatures,
    GQAFilemap,
    GQASplit,
    GQAVersion,
)
from ...datasets.gqa import (
    GQA,
    GQAImages,
    GQAObjects,
    GQAQuestions,
    GQASceneGraphs,
    GQASpatial,
)
from ..preprocessing import QuestionTransformer, SceneGraphTransformer
from .preprocessing_factory import PreprocessorCollection


@dataclass(frozen=True)
class DatasetCollection:
    """Wrapper class for storing a train, val and test dataset tuple."""

    train: Dataset
    val: Dataset
    test: Dataset


class DatasetFactory:
    """Factory class for creating datasets given a configuration object."""

    def __init__(self) -> None:
        """Initialise the dataset factory."""
        self._factory_methods = {
            DatasetName.GQA: DatasetFactory._create_gqa,
            DatasetName.CLEVR: DatasetFactory._create_clevr,
        }

    def create(
        self, config: Config
    ) -> Tuple[DatasetCollection, PreprocessorCollection]:
        """Create a dataset from a given config."""
        return self._factory_methods[config.dataset.name](config)

    @staticmethod
    def _create_clevr(
        config: Config,
    ) -> Tuple[DatasetCollection, PreprocessorCollection]:
        if not isinstance(config.dataset, CLEVRDatasetConfig):
            raise ValueError(
                f"Param {config.dataset=} must be of type",
                f"{CLEVRDatasetConfig.__name__}.",
            )
        raise NotImplementedError()

    @staticmethod
    def _create_gqa(
        config: Config,
    ) -> Tuple[DatasetCollection, PreprocessorCollection]:
        # pylint: disable=too-many-branches,too-many-locals

        if not isinstance(config.dataset, GQADatasetConfig):
            raise ValueError(
                f"Param {config.dataset=} must be of type {GQADatasetConfig.__name__}."
            )

        datasets: List[GQA] = []

        for split_key, subset in {
            "train": config.model.data.train,
            "val": config.model.data.val,
            "test": config.model.data.test,
        }.items():

            if subset.split not in [split.value for split in iter(GQASplit)]:
                raise ValueError("Invalid split string.")

            if subset.version not in [version.value for version in iter(GQAVersion)]:
                raise ValueError("Invalid version string.")

            questions = None
            images = None
            objects = None
            spatial = None
            scene_graphs = None

            if GQAFeatures.QUESTIONS.value not in [
                feat.name for feat in config.model.data.features
            ]:
                raise ValueError(
                    f'List of features must contain "{GQAFeatures.QUESTIONS.value}"'
                )

            for feature in config.model.data.features:
                if feature.name not in [feature.value for feature in iter(GQAFeatures)]:
                    raise ValueError("Invalid feature string.")

                # By default, use unprocessed data.
                filemap = config.dataset.filemap

                # If an artifact is specified, use it for that feature instead.
                if feature.artifact is not None:
                    try:
                        # Load artifact
                        artifact = wandb.run.use_artifact(feature.artifact)
                        artifact_dir = Path(artifact.download())
                        filemap = GQAFilemap(root=artifact_dir)
                        # Load preprocessor for this feature
                        with open(
                            artifact_dir / "preprocessors.json", "r"
                        ) as json_file:
                            preprocessors = jsons.load(
                                json.load(json_file),
                                PreprocessorCollection,
                            )
                    except (wandb.CommError, AttributeError):
                        print(
                            "Could not load artifact for feature",
                            f'"{feature.name}", using raw dataset instead.',
                        )
                    except FileNotFoundError as ex:
                        raise ValueError(
                            "Could not load preprocessor for feature",
                            f'"{feature.name}" in artifact "{feature.artifact}".',
                        ) from ex

                if feature.name == GQAFeatures.QUESTIONS.value:
                    questions = GQAQuestions(
                        filemap,
                        GQASplit(subset.split),
                        GQAVersion(subset.version),
                        transform=QuestionTransformer(),
                    )
                elif feature.name == GQAFeatures.IMAGES.value:
                    images = GQAImages(filemap, transform=None)
                elif feature.name == GQAFeatures.SCENE_GRAPHS.value:
                    if GQASplit(subset.split) in (GQASplit.TRAIN, GQASplit.VAL):
                        scene_graphs = GQASceneGraphs(
                            filemap,
                            GQASplit(subset.split),
                            transform=SceneGraphTransformer(),
                        )
                elif feature.name == GQAFeatures.SPATIAL.value:
                    spatial = GQASpatial(filemap)
                elif feature.name == GQAFeatures.OBJECTS.value:
                    objects = GQAObjects(filemap)
                else:
                    raise NotImplementedError()

            datasets.append(
                GQA(
                    GQASplit(subset.split),
                    GQAVersion(subset.version),
                    questions=questions,
                    images=images,
                    objects=objects,
                    spatial=spatial,
                    scene_graphs=scene_graphs,
                )
            )

        return (
            DatasetCollection(train=datasets[0], val=datasets[1], test=datasets[2]),
            preprocessors,
        )
