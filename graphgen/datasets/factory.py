"""Tools for creating datasets given configuration objects."""
import torch.utils.data

from ..config import Config
from ..config.clevr import CLEVRDatasetConfig
from ..config.dataset import DatasetName
from ..config.gqa import GQADatasetConfig, GQAFeatures
from ..utilities.preprocessing import GQAQuestionPreprocessor, custom_transform
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
        questions = GQAQuestions(
            dataset_config.filemap,
            dataset_config.split,
            dataset_config.version,
            cache=config.cache,
            preprocessor=GQAQuestionPreprocessor(),
            transform=custom_transform,
        )
        images = None
        objects = None
        spatial = None
        scene_graphs = None

        for feature in config.dataset.features:
            if feature == GQAFeatures.IMAGES:
                images = GQAImages(dataset_config.filemap)
            elif feature == GQAFeatures.OBJECTS:
                objects = GQAObjects(dataset_config.filemap)
            elif feature == GQAFeatures.SPATIAL:
                spatial = GQASpatial(dataset_config.filemap)
            elif feature == GQAFeatures.SCENE_GRAPHS:
                spatial = GQASceneGraphs(
                    dataset_config.filemap, dataset_config.split, cache=config.cache
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
