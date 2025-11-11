from abc import ABC, abstractmethod
from infrastructure.frameworks.model_inference_maker import ModelInferenceMaker
from typing import Any, Dict, List

class ModelService(ABC):
    """
    This interface defines the contract for managing models in the application.
    """
    @abstractmethod
    def get_model_inference_maker(self, entity_set_id: str, model_id: str) -> ModelInferenceMaker:
        """
        Returns an appropriate instance of inference maker for the specified model.
        :param entity_set_id: The ID of the entity set for which the model is requested.
        :param model_id: The ID of the model to be used for inference.
        :return: An instance of ModelInferenceMaker for the specified model.
        """
        pass

    @abstractmethod
    def list_models(self, entity_set_id: str) -> Dict[str, str]:
        """
        List available models for an entity set
        :param entity_set_id: The ID of the entity set for which models are listed.
        :return: A dictionary mapping model IDs to model types.
        """
        pass

    @abstractmethod
    def get_entity_set_labels(self, entity_set_id: str) -> List[str]:
        """
        Get the labels for the entities in the specified entity set.
        :param entity_set_id: The ID of the entity set for which labels are requested.
        :return: A list of entity labels.
        """
        pass

    @abstractmethod
    def get_model_config(self, entity_set_id: str, model_id: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model
        :param entity_set_id: The ID of the entity set for which the model configuration is requested.
        :param model_id: The ID of the model for which the configuration is requested.
        :return: A dictionary containing the model configuration.
        """
        pass

    @abstractmethod
    def reload_model_registry(self) -> None:
        """
        Reload model registry from configuration.
        This method should be called to refresh the model registry, typically after configuration changes.
        :return: None
        """
        pass