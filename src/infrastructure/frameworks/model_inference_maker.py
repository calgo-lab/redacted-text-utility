from abc import ABC, abstractmethod
from typing import Any

from infrastructure.frameworks.model_loader import ModelLoader


class ModelInferenceMaker(ABC):
    """
    Interface for model inference makers.
    This interface defines the contract for making inferences using loaded models.
    Each inference maker should implement the infer method to return the inference result.
    """

    def __init__(self, model_loader: ModelLoader):
        """
        Initialize the ModelInferenceMaker with a ModelLoader instance.
        
        :param model_loader: An instance of ModelLoader to load the model.
        """
        self.model_loader = model_loader

    @abstractmethod
    def infer(self, input_text: str, **kwargs) -> Any:
        """
        Make inference using the loaded model.
        
        :param input_text: The input text for which inference is to be made.
        :param **kwargs: Additional keyword arguments for inference.
        :return: The inference result.
        """
        pass