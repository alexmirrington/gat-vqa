"""Tools for creating trainable datasets given configuration objects."""
import json
from pathlib import Path
from typing import List, Tuple, Union

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
from ...config.prediction import PredictionConfig
from ...config.training import TrainingConfig
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
from .runner_factory import RunnerFactory


class DatasetFactory:
    """Factory class for creating datasets given a configuration object."""

    def __init__(self, training: bool = True) -> None:
        """Initialise the dataset factory."""
        self._factory_methods = {
            DatasetName.GQA: self._create_gqa,
            DatasetName.CLEVR: self._create_clevr,
        }
        self.training = training

    def create(
        self, config: Config
    ) -> Tuple[DatasetCollection, PreprocessorCollection]:
        """Create a dataset from a given config."""
        return self._factory_methods[config.dataset.name](config)

    def _create_clevr(
        self,
        config: Config,
    ) -> Tuple[DatasetCollection, PreprocessorCollection]:
        # pylint: disable=no-self-use
        if not isinstance(config.dataset, CLEVRDatasetConfig):
            raise ValueError(
                f"Param {config.dataset} must be of type",
                f"{CLEVRDatasetConfig.__name__}.",
            )
        raise NotImplementedError()

    def _create_gqa(
        self,
        config: Config,
    ) -> Tuple[DatasetCollection, PreprocessorCollection]:
        # pylint: disable=too-many-branches,too-many-locals,too-many-statements

        if not isinstance(config.dataset, GQADatasetConfig):
            raise ValueError(
                f"Param {config.dataset} must be of type {GQADatasetConfig.__name__}."
            )

        datasets: List[GQA] = []

        split_map = (
            {
                "train": config.training.data.train,
                "val": config.training.data.val,
                "test": config.training.data.test,
            }
            if self.training
            else {
                "train": config.prediction.data.train,
                "val": config.prediction.data.val,
                "test": config.prediction.data.test,
            }
        )

        cfg: Union[TrainingConfig, PredictionConfig] = (
            config.training if self.training else config.prediction
        )

        for split_key, subset in split_map.items():

            if subset.split not in [split.value for split in iter(GQASplit)]:
                raise ValueError("Invalid split string.")

            if subset.version not in [version.value for version in iter(GQAVersion)]:
                raise ValueError("Invalid version string.")

            questions = None
            images = None
            objects = None
            spatial = None
            scene_graphs = None

            for feature in cfg.data.features:
                if feature.name not in [feature.value for feature in iter(GQAFeatures)]:
                    raise ValueError("Invalid feature string.")

                # By default, use unprocessed data.
                filemap = config.dataset.filemap

                # If an artifact is specified, use it for that feature instead.
                if feature.artifact is not None:
                    artifact_dir = None

                    # Try and load artifact from wandb
                    try:
                        # Load artifact from wandb
                        artifact = wandb.run.use_artifact(feature.artifact)
                        artifact_dir = Path(artifact.download())
                    except (wandb.CommError, AttributeError):
                        pass

                    # Try and load artifact from local directory
                    if artifact_dir is None:
                        artifact_dir = Path(feature.artifact)
                        if not artifact_dir.is_dir():
                            artifact_dir = None

                    # Assume we have a valid artifact directory now
                    if artifact_dir is not None:
                        try:
                            filemap = GQAFilemap(root=artifact_dir)
                            # Load preprocessor for this feature
                            with open(
                                artifact_dir / "preprocessors.json", "r"
                            ) as json_file:
                                preprocessors = jsons.load(
                                    json.load(json_file),
                                    PreprocessorCollection,
                                )
                            print(f'Loaded data for feature "{feature.name}"')
                        except FileNotFoundError as ex:
                            raise ValueError(
                                "Could not load preprocessor for feature",
                                f'"{feature.name}" from path/artifact',
                                f"{feature.artifact}.",
                            ) from ex
                    else:
                        raise ValueError(f"Invalid artifact '{feature.artifact}'")

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
                                num_objects=len(
                                    preprocessors.scene_graphs.object_to_index
                                ),
                                num_relations=len(
                                    preprocessors.scene_graphs.rel_to_index
                                ),
                                num_attributes=len(
                                    preprocessors.scene_graphs.attr_to_index
                                ),
                                graph=config.model.scene_graph.graph,
                                embeddings=RunnerFactory.build_embeddings(
                                    config.model.scene_graph.embedding,
                                    list(
                                        preprocessors.scene_graphs.object_to_index.keys()  # noqa: B950
                                    )
                                    + list(
                                        preprocessors.scene_graphs.rel_to_index.keys()
                                    )
                                    + list(
                                        preprocessors.scene_graphs.attr_to_index.keys()
                                    ),
                                )
                                if not config.model.scene_graph.embedding.trainable
                                else None,
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
            DatasetCollection(
                train=datasets[0],
                val=datasets[1],
                test=datasets[2],
                images=GQAImages(config.dataset.filemap),
            ),
            preprocessors,
        )
