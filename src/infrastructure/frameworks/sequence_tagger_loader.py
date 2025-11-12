from core.exceptions import ModelLoadError
from core.logging import get_logger
from flair.models import SequenceTagger
from infrastructure.frameworks.cached_model_loader import CachedModelLoader
from pathlib import Path

import flair


class SequenceTaggerLoader(CachedModelLoader):
    """
    Responsible for loading a Flair SequenceTagger model - given a model_path.
    """
    def __init__(self, 
                 model_name_or_path: str, 
                 cache_root: Path, 
                 loading_strategy: str = "local_disk_storage"):
        """
        :param model_name_or_path: The path or name of the Flair model to load.
        :param loading_strategy: The strategy to use for loading the model (e.g., local_disk_storage).
        """
        super().__init__(model_name_or_path, loading_strategy)
        self.logger = get_logger(__name__)
        self._cache_root: Path = cache_root

    def _load_model(self) -> SequenceTagger:
        """
        Load the Flair SequenceTagger model and return the model object.
        :return: The loaded SequenceTagger model object.
        """
        flair.cache_root = self._cache_root
        try:
            if self.loading_strategy == "local_disk_storage":
                model_file_path: Path = Path(self.model_name_or_path) / "model.pt"
                if not model_file_path.exists():
                    raise FileNotFoundError(f"Model file not found at configured location: {model_file_path}")
                return SequenceTagger.load(model_file_path)
            elif self.loading_strategy == "huggingface_hub":
                return SequenceTagger.load(self.model_name_or_path)
        except Exception as e:
            self.logger.error(f"Failed to load Flair model for '{self.model_name_or_path}': {e}")
            raise ModelLoadError(str(e)) from e