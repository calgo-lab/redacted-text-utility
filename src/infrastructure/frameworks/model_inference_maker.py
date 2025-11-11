from abc import ABC, abstractmethod

from infrastructure.frameworks.model_loader import ModelLoader
from typing import Any

class ModelInferenceMaker(ABC):
    """
    Interface for model inference makers.
    This interface defines the contract for making inferences using loaded models.
    Each inference maker should implement the infer method to return the inference result.
    """

    def __init__(self, model_loader: ModelLoader):
        """
        :param model_loader: The ModelLoader instance with the model object.
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