"""Tools for preprocessing datasets given a pipeline in the form of a config object."""
import json
from pathlib import Path
from typing import Any, Callable

import jsons
import wandb

from ...config import Config
from ...config.dataset import DatasetName
from ...config.gqa import (
    GQADatasetConfig,
    GQAFeatures,
    GQAFilemap,
    GQASplit,
    GQAVersion,
)
from ...datasets.gqa import GQAQuestions, GQASceneGraphs
from ...datasets.utilities import KeyedDataset
from ..preprocessing import (
    GQAQuestionPreprocessor,
    GQASceneGraphPreprocessor,
    PreprocessorCollection,
)


class PreprocessingFactory:
    """Factory class for preprocessing datasets given a pipeline in the form \
    of a configuration object."""

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
    def _apply_preprocessor(
        source: KeyedDataset,
        preprocessor: Callable[[Any], Any],
        cache: Path,
        chunks: int,
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

    @staticmethod
    def _process_clevr(config: Config) -> None:
        raise NotImplementedError()

    @staticmethod
    def _process_gqa(config: Config) -> None:
        if not isinstance(config.dataset, GQADatasetConfig):
            raise ValueError(
                f"Param {config.dataset=} must be of type {GQADatasetConfig.__name__}."
            )

        preprocessors = PreprocessorCollection(
            questions=GQAQuestionPreprocessor(),
            scene_graphs=GQASceneGraphPreprocessor(),
        )

        root = config.preprocessing.cache.root / wandb.run.id
        if not root.exists():
            root.mkdir()
        new_filemap = GQAFilemap(root=root)

        for item in config.preprocessing.pipeline:

            if item.split not in [split.value for split in iter(GQASplit)]:
                raise ValueError("Invalid split string.")

            if item.feature not in [feat.value for feat in iter(GQAFeatures)]:
                raise ValueError("Invalid feature string.")

            # Ensure version is specified for GQA questions
            if item.feature == GQAFeatures.QUESTIONS.value and item.version not in [
                version.value for version in iter(GQAVersion)
            ]:
                raise ValueError("Invalid version string.")

            print(f"processing {item}.")

            if item.feature == GQAFeatures.QUESTIONS.value:
                questions = GQAQuestions(
                    config.dataset.filemap,
                    GQASplit(item.split),
                    GQAVersion(item.version),
                    transform=None,
                )
                chunks = len(questions.chunk_sizes)
                PreprocessingFactory._apply_preprocessor(
                    source=questions,
                    preprocessor=preprocessors.questions,
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
                scene_graphs = GQASceneGraphs(
                    config.dataset.filemap,
                    GQASplit(item.split),
                    transform=None,
                )
                chunks = len(scene_graphs.chunk_sizes)
                PreprocessingFactory._apply_preprocessor(
                    source=scene_graphs,
                    preprocessor=preprocessors.scene_graphs,
                    cache=new_filemap.scene_graph_path(GQASplit(item.split)),
                    chunks=chunks,
                )

        # Dump preprocessors
        with open(root / "preprocessors.json", "w") as json_file:
            json.dump(jsons.dump(preprocessors, strip_privates=True), json_file)

        # Log artifact
        artifact = wandb.Artifact(
            config.preprocessing.cache.artifact,
            type="dataset",
            metadata=jsons.dump(config.preprocessing),
        )
        artifact.add_dir(new_filemap.root)
        wandb.run.log_artifact(artifact)
