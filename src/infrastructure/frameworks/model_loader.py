from abc import ABC, abstractmethod
from typing import Any


class ModelLoader(ABC):
    """
    Interface for all model loaders.
    This interface defines the contract for loading models in different frameworks.
    Each loader should implement the load method to return the model object.
    """

    def __init__(self, 
                 model_name_or_path: str, 
                 loading_strategy: str = "local_disk_storage"):
        """
        Initialize the ModelLoader with model name or path and loading strategy.
        
        :param model_name_or_path: The name or path of the model to be loaded.
        :param loading_strategy: The strategy to use for loading the model. 
                                 Defaults to "local_disk_storage".
        """
        self.model_name_or_path = model_name_or_path
        self.loading_strategy = loading_strategy

    @abstractmethod
    def load(self) -> Any:
        """
        Load the model and return the model object.
        
        :return: The loaded model object.
        """
        pass