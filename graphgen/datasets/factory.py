"""Tools for creating datasets given configuration objects."""
from typing import List, Tuple

from torch.utils.data import Dataset

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

        question_preprocessor = GQAQuestionPreprocessor()
        datasets: List[GQA] = []
        for subset_config in (
            config.dataset.train,
            config.dataset.val,
            config.dataset.test,
        ):
            # Parse preprocessing pipeline caches
            # TODO move cache path generation to preprocessing config object
            # TODO implement cache levels ["global", "branch", "commit"]
            caches = {
                item.feature: (
                    config.preprocessing.cache
                    / "global"
                    / config.dataset.name.value
                    / item.feature
                    / subset_config.split.value
                    / subset_config.version.value
                )
                for item in config.preprocessing.pipeline
            }

            questions = GQAQuestions(
                config.dataset.filemap,
                subset_config.split,
                subset_config.version,
                cache=caches[GQAFeatures.QUESTIONS.value],
                preprocessor=question_preprocessor,
                transform=QuestionTransformer(),
            )

            # Freeze question preprocessor for val and test so vocab is
            # not updated
            question_preprocessor.frozen = True

            images = None
            objects = None
            spatial = None
            scene_graphs = None

            for feature in config.dataset.features:
                if feature == GQAFeatures.QUESTIONS:
                    continue
                if feature == GQAFeatures.IMAGES:
                    images = GQAImages(config.dataset.filemap)
                elif feature == GQAFeatures.OBJECTS:
                    objects = GQAObjects(config.dataset.filemap)
                elif feature == GQAFeatures.SPATIAL:
                    spatial = GQASpatial(config.dataset.filemap)
                elif feature == GQAFeatures.SCENE_GRAPHS:
                    spatial = GQASceneGraphs(
                        config.dataset.filemap,
                        subset_config.split,
                        cache=caches[GQAFeatures.SCENE_GRAPHS.value],
                    )
                else:
                    raise NotImplementedError()
            gqa = GQA(
                questions,
                images=images,
                objects=objects,
                spatial=spatial,
                scene_graphs=scene_graphs,
            )
            datasets.append(gqa)
        # Keep mypy happy, it doesn't like `tuple(datasets)` ¯\_(ツ)_/¯
        return (datasets[0], datasets[1], datasets[2])
