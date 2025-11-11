from abc import abstractmethod
from infrastructure.frameworks.model_loader import ModelLoader
from typing import Any


class CachedModelLoader(ModelLoader):
    """
    Base class ensuring caching across an app run.
    """
    _model = None
    
    def __init__(self, 
                 model_name_or_path: str, 
                 loading_strategy: str = "local_disk_storage"):
        """
        :param model_name_or_path: The path or name of the model to load.
        :param loading_strategy: The strategy to use for loading the model (e.g., local_disk_storage).
        """
        super().__init__(model_name_or_path, loading_strategy)

    def load(self) -> Any:
        """
        Load the model and return the model object if not already loaded.
        :return: The loaded model object.
        """
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @abstractmethod
    def _load_model(self) -> Any:
        """
        This method should be implemented by subclasses to define how the model is loaded.
        :return: The loaded model object.
        """
        pass