"""Tools for creating trainable datasets given configuration objects."""
import json
from pathlib import Path
from typing import List, Tuple

import jsons
import wandb

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
from ...config.model import GCNModelConfig, MACMultiGCNModelConfig
from ...datasets.gqa import (
    GQA,
    GQAImages,
    GQAObjects,
    GQAQuestions,
    GQASceneGraphs,
    GQASpatial,
)
from ..preprocessing import (
    DatasetCollection,
    ObjectTransformer,
    PreprocessorCollection,
    QuestionTransformer,
    SceneGraphTransformer,
)


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
                f"Param {config.dataset} must be of type",
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
                f"Param {config.dataset} must be of type {GQADatasetConfig.__name__}."
            )

        datasets: List[GQA] = []

        for split_key, subset in {
            "train": config.training.data.train,
            "val": config.training.data.val,
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

            for feature in config.training.data.features:
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
                            transform=SceneGraphTransformer(
                                embedding=config.model.scene_graph.embedding
                                if isinstance(config.model, MACMultiGCNModelConfig)
                                and isinstance(config.model.scene_graph, GCNModelConfig)
                                else None,  # TODO parameterise in config
                                num_objects=len(
                                    preprocessors.scene_graphs.object_to_index
                                ),
                                num_relations=len(
                                    preprocessors.scene_graphs.rel_to_index
                                ),
                                num_attributes=len(
                                    preprocessors.scene_graphs.attr_to_index
                                ),
                            ),
                        )
                elif feature.name == GQAFeatures.SPATIAL.value:
                    spatial = GQASpatial(filemap)
                elif feature.name == GQAFeatures.OBJECTS.value:
                    objects = GQAObjects(filemap, transform=ObjectTransformer())
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
            DatasetCollection(train=datasets[0], val=datasets[1]),
            preprocessors,
        )
