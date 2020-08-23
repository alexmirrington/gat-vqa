"""Tools for creating trainable datasets given configuration objects."""
import json
from pathlib import Path
from typing import Tuple

import jsons
import wandb
from torch.utils.data import Dataset

from ..config import Config
from ..config.clevr import CLEVRDatasetConfig
from ..config.dataset import DatasetName
from ..config.gqa import GQADatasetConfig, GQAFeatures, GQAFilemap, GQASplit, GQAVersion
from ..datasets.utilities import KeyedDataset
from ..utilities.preprocessing import (
    GQAQuestionPreprocessor,
    Preprocessor,
    QuestionTransformer,
)
from .gqa import GQA, GQAQuestions


class DatasetFactory:
    """Factory class for creating datasets given a configuration object."""

    def __init__(self) -> None:
        """Initialise the dataset factory."""
        self._factory_methods = {
            DatasetName.GQA: DatasetFactory._create_gqa,
            DatasetName.CLEVR: DatasetFactory._create_clevr,
        }

    def create(self, config: Config) -> Tuple[Dataset, Dataset, Dataset]:
        """Create a dataset from a given config."""
        return self._factory_methods[config.dataset.name](config)

    @staticmethod
    def _create_clevr(config: Config) -> Tuple[Dataset, Dataset, Dataset]:
        if not isinstance(config.dataset, CLEVRDatasetConfig):
            raise ValueError(
                f"Param {config.dataset=} must be of type",
                f"{CLEVRDatasetConfig.__name__}.",
            )
        raise NotImplementedError()

    @staticmethod
    def _create_gqa(config: Config) -> Tuple[GQA, GQA, GQA]:
        if not isinstance(config.dataset, GQADatasetConfig):
            raise ValueError(
                f"Param {config.dataset=} must be of type {GQADatasetConfig.__name__}."
            )

        artifact = wandb.run.use_artifact(config.model.data.artifact)
        artifact_dir = Path(artifact.download())

        new_filemap = GQAFilemap(root=artifact_dir)

        # TODO Fallback to unprocessed data if no data exists
        # in preprocessed data

        datasets = []

        for subset in (
            config.model.data.train,
            config.model.data.val,
            config.model.data.test,
        ):

            if subset.split not in [split.value for split in iter(GQASplit)]:
                raise ValueError("Invalid split string.")

            if subset.version not in [version.value for version in iter(GQAVersion)]:
                raise ValueError("Invalid version string.")

            questions = GQAQuestions(
                new_filemap,
                GQASplit(subset.split),
                GQAVersion(subset.version),
                transform=QuestionTransformer(),
            )

            images = None
            objects = None
            spatial = None
            scene_graphs = None

            for feature in config.dataset.features:
                if feature != GQAFeatures.QUESTIONS:
                    raise NotImplementedError()
            gqa = GQA(
                questions,
                images=images,
                objects=objects,
                spatial=spatial,
                scene_graphs=scene_graphs,
            )
            datasets.append(gqa)

        return (datasets[0], datasets[1], datasets[2])


def process(
    source: KeyedDataset, preprocessor: Preprocessor, cache: Path, chunks: int
) -> None:
    """Process a `source` dataset and save it at `cache`."""
    is_file = False
    if cache.suffix != "":
        is_file = True
        if not cache.parent.exists():
            cache.parent.mkdir(parents=True)
    elif not cache.exists():
        cache.mkdir(parents=True)

    # Preprocess
    keys = []
    data = []
    chunk_size = len(source) // chunks
    for idx, key in enumerate(source.keys()):
        keys.append(key)
        data.append(source[source.key_to_index(key)])
        if idx % chunk_size == chunk_size - 1 or idx == len(source) - 1:
            preprocessed_data = preprocessor(data)
            # Save to file
            path = cache if is_file else cache / f"{idx // chunk_size}.json"
            with open(path, "w") as json_file:
                json.dump(dict(zip(keys, preprocessed_data)), json_file)
            del preprocessed_data
            keys = []
            data = []

    # # Dump preprocessor object public fields so we can inspect information
    # # about the preprocessed dataset.
    # preprocessor = jsons.dump(preprocessor, strip_privates=True)
    # metadata = {"preprocessor": preprocessor}
    # with open(cache / "meta.json", "w") as json_file:
    #     json.dump(metadata, json_file)


class PreprocessingFactory:
    """Factory class for preprocessing datasets given a configuration object."""

    def __init__(self) -> None:
        """Initialise the preprocessing factory."""
        self._factory_methods = {
            DatasetName.GQA: PreprocessingFactory._process_gqa,
            DatasetName.CLEVR: PreprocessingFactory._process_clevr,
        }

    def process(self, config: Config) -> None:
        """Create a dataset from a given config."""
        return self._factory_methods[config.dataset.name](config)

    @staticmethod
    def _process_clevr(config: Config) -> None:
        raise NotImplementedError()

    @staticmethod
    def _process_gqa(config: Config) -> None:
        if not isinstance(config.dataset, GQADatasetConfig):
            raise ValueError(
                f"Param {config.dataset=} must be of type {GQADatasetConfig.__name__}."
            )

        question_preprocessor = GQAQuestionPreprocessor()

        root = config.preprocessing.cache.root / wandb.run.id
        if not root.exists():
            root.mkdir()
        new_filemap = GQAFilemap(root=root)

        for item in config.preprocessing.pipeline:

            if item.split not in [split.value for split in iter(GQASplit)]:
                raise ValueError("Invalid split string.")

            if item.version not in [version.value for version in iter(GQAVersion)]:
                raise ValueError("Invalid version string.")

            if item.feature not in [feat.value for feat in iter(GQAFeatures)]:
                raise ValueError("Invalid feature string.")

            if item.feature == GQAFeatures.QUESTIONS.value:
                questions = GQAQuestions(
                    config.dataset.filemap,
                    GQASplit(item.split),
                    GQAVersion(item.version),
                    transform=None,
                )
                chunks = len(questions.chunk_sizes)
                process(
                    source=questions,
                    preprocessor=question_preprocessor,
                    cache=new_filemap.question_path(
                        GQASplit(item.split),
                        GQAVersion(item.version),
                        chunked=(chunks > 1),
                    ),
                    chunks=chunks,
                )
            elif item.feature == GQAFeatures.IMAGES.value:
                raise NotImplementedError()
            elif item.feature == GQAFeatures.OBJECTS.value:
                raise NotImplementedError()
            elif item.feature == GQAFeatures.SPATIAL.value:
                raise NotImplementedError()
            elif item.feature == GQAFeatures.SCENE_GRAPHS.value:
                raise NotImplementedError()
            else:
                raise NotImplementedError()

        artifact = wandb.Artifact(
            config.preprocessing.cache.artifact,
            type="dataset",
            metadata=jsons.dump(config.preprocessing),
        )
        artifact.add_dir(new_filemap.root)
        wandb.run.log_artifact(artifact)
