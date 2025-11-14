from typing import Any, Dict, Optional


class BaseException(Exception):
    """
    Base exception class that works as a parent for all custom exceptions.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the BaseException with a message and optional details.
        
        :param message: The exception message.
        :param details: Optional dictionary containing additional details about the exception.
        """
        super().__init__(message)
        self._message = message
        self._details = details if details is not None else dict()
    
    @property
    def message(self) -> str:
        """
        Get the exception message.
        """
        return self._message

    @property
    def details(self) -> Dict[str, Any]:
        """
        Get additional exception details.
        """
        return self._details

class ResourceNotFoundException(BaseException):
    """
    Raised when a requested resource is not found
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the ResourceNotFoundException with a message and optional details.
        
        :param message: The exception message.
        :param details: Optional dictionary containing additional details about the exception.
        """
        super().__init__(message, details)

class ConfigurationException(BaseException):
    """
    Raised when configuration is invalid or missing
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the ConfigurationException with a message and optional details.
        
        :param message: The exception message.
        :param details: Optional dictionary containing additional details about the exception.
        """
        super().__init__(message, details)

class PredictionException(BaseException):
    """
    Raised when prediction fails.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the PredictionException with a message and optional details.
        
        :param message: The exception message.
        :param details: Optional dictionary containing additional details about the exception.
        """
        super().__init__(message, details)


class EntitySetNotFoundException(ResourceNotFoundException):
    """
    Raised when an entity set is not found in the registry
    """

    def __init__(self, entity_set_id: str):
        """
        Initialize the EntitySetNotFoundException with the entity set ID.

        :param entity_set_id: The ID of the entity set that was not found.
        """
        super().__init__(
            message=f"Entity set '{entity_set_id}' not found",
            details={"entity_set_id": entity_set_id}
        )

class ModelNotFoundException(ResourceNotFoundException):
    """
    Raised when a model is not found in the registry for the given entity set id and model id
    """

    def __init__(self, entity_set_id: str, model_id: str):
        """
        Initialize the ModelNotFoundException with the entity set ID and model ID.

        :param entity_set_id: The ID of the entity set.
        :param model_id: The ID of the model that was not found.
        """
        super().__init__(
            message=f"Model '{model_id}' not found for entity set '{entity_set_id}'",
            details={"entity_set_id": entity_set_id, "model_id": model_id}
        )

class UnsupportedModelLoadingStrategyException(ConfigurationException):
    """
    Raised when an unsupported model loading strategy is encountered
    """

    def __init__(self, entity_set_id: str, model_id: str, strategy: str):
        """
        Initialize the UnsupportedModelLoadingStrategyException with the entity set ID, model ID, and strategy.
        
        :param entity_set_id: The ID of the entity set.
        :param model_id: The ID of the model.
        :param strategy: The unsupported loading strategy.
        """
        super().__init__(
            message=f"Unsupported model loading strategy '{strategy}' for model '{model_id}' in entity set '{entity_set_id}'",
            details={
                "entity_set_id": entity_set_id,
                "model_id": model_id,
                "strategy": strategy
            }
        )

class UnsupportedModelImplTypeException(ConfigurationException):
    """
    Raised when an unsupported model impl type is encountered
    """

    def __init__(self, entity_set_id: str, model_id: str, model_impl: str):
        """
        Initialize the UnsupportedModelImplTypeException with the entity set ID, model ID, and model implementation type.
        
        :param entity_set_id: The ID of the entity set.
        :param model_id: The ID of the model.
        :param model_impl: The unsupported model implementation type.
        """
        super().__init__(
            message=f"Unsupported model impl type: '{model_impl}' for model '{model_id}' in entity set '{entity_set_id}'",
            details={
                "entity_set_id": entity_set_id,
                "model_id": model_id,
                "model_impl": model_impl
            }
        )

class ModelLoadException(ConfigurationException):
    """
    Raised when a model fails to load
    """

    def __init__(self, model_name_or_path: str):
        """
        Initialize the ModelLoadException with the model name or path.
        
        :param model_name_or_path: The name or path of the model that failed to load.
        """
        super().__init__(
            message=f"Failed to load model for '{model_name_or_path}'",
            details={"model_name_or_path": model_name_or_path}
        )

class UnsupportedOperationForModelException(PredictionException):
    """
    Raised when model_type is mismatched for respective operation
    """
    
    def __init__(self, entity_set_id: str, model_id: str, model_type: str, required_model_type: str):
        """
        Initialize the UnsupportedOperationForModelException with the entity set ID, model ID, current model type, and required model type.
        
        :param entity_set_id: The ID of the entity set.
        :param model_id: The ID of the model.
        :param model_type: The current model type.
        :param required_model_type: The required model type for the operation.
        """
        super().__init__(
            message=f"Unsupported operation requested, required model_type '{required_model_type}', but found model_type '{model_type}' for model '{model_id}' in entity set '{entity_set_id}'",
            details={
                "entity_set_id": entity_set_id,
                "model_id": model_id,
                "model_type": model_type,
                "required_model_type": required_model_type
            }
        )