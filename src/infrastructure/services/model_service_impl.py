from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version
from pathlib import Path
from typing import Any, Dict, List, Tuple

from config_handlers.entity_set_models_config_handler import EntitySetModelsConfigHandler
from config_handlers.frameworks_config_handler import FrameworksConfigHandler
from core.exceptions import (
    EntitySetNotFoundException,
    ModelNotFoundException,
    UnsupportedModelLoadingStrategyException,
    UnsupportedModelImplTypeException
)
from core.logging import get_logger
from infrastructure.frameworks.model_inference_maker import ModelInferenceMaker
from infrastructure.frameworks.model_loader import ModelLoader
from infrastructure.frameworks.sequence_tagger_inference_maker import SequenceTaggerInferenceMaker
from infrastructure.frameworks.sequence_tagger_loader import SequenceTaggerLoader
from infrastructure.services.model_service import ModelService
from utils.project_utils import ProjectUtils

import importlib.metadata


class ModelServiceImpl(ModelService):
    """
    Implementation of the ModelService interface.
    This service manages the loading and retrieval of models based on entity sets and model IDs.
    It uses different configuration handlers to load models according to the configuration.
    """

    _models_registry: Dict[str, Dict[str, Tuple[ModelLoader, ModelInferenceMaker]]] = dict()
    
    def __init__(self):
        """
        Initialize the ModelServiceImpl.
        This constructor loads the model registry.
        """
        self.logger = get_logger(__name__)
        self._entity_set_models_config_handler = EntitySetModelsConfigHandler.load_from_file()
        self._frameworks_config_handler = FrameworksConfigHandler.load_from_file(
            replacements={
                "PROJECT_ROOT": str(ProjectUtils.get_project_root())
            }
        )
        self._load_model_registry()
    
    def _load_model_registry(self) -> None:
        """
        Load the model registry from the entity set models configuration.
        This method initializes the _models_registry dictionary with models from the entity set models configuration.
        It clears any existing models and reloads them based on the current configuration.

        :return: None
        """
        self._models_registry.clear()
        for entity_set_cfg in self._entity_set_models_config_handler.entity_sets:
            entity_set_id = entity_set_cfg.entity_set_id
            self._models_registry[entity_set_id] = dict()
            for model_cfg in entity_set_cfg.supported_models:
                model_loader, model_inference_maker = self._load(entity_set_cfg, model_cfg)
                if model_loader and model_inference_maker:
                    self._models_registry[entity_set_id][model_cfg.model_id] = (model_loader, model_inference_maker)

    def _load(self, entity_set_cfg, model_cfg) -> Tuple[ModelLoader, ModelInferenceMaker]:
        """
        Load a model based on the entity set configuration and model configuration.
        
        :param entity_set_cfg: The configuration for the entity set.
        :param model_cfg: The configuration for the model.
        :return: A tuple containing the ModelLoader and ModelInferenceMaker instances.
        """
        model_loader: ModelLoader = None
        model_inference_maker: ModelInferenceMaker = None
        
        requirements_satisfied = self._check_requirements(model_cfg.model_system_requirements)
        if requirements_satisfied:
            self.logger.info(f"Requirements satisfied for model '{model_cfg.model_id}' in entity set '{entity_set_cfg.entity_set_id}'")
        else:
            self.logger.warning(f"Requirements not satisfied, skipping model loading in this instance for model '{model_cfg.model_id}' in entity set '{entity_set_cfg.entity_set_id}'")
            return None, None
        
        if model_cfg.model_loading_strategy not in ["local_disk_storage", "huggingface_hub"]:
            self.logger.error(f"Unsupported model loading strategy {model_cfg.model_loading_strategy} for model '{model_cfg.model_id}' in entity set '{entity_set_cfg.entity_set_id}'")
            raise UnsupportedModelLoadingStrategyException(entity_set_cfg.entity_set_id, model_cfg.model_id, model_cfg.model_loading_strategy)
        
        if model_cfg.model_impl not in ["SequenceTagger"]:
            self.logger.error(f"Unsupported model impl type {model_cfg.model_impl} for model '{model_cfg.model_id}' in entity set '{entity_set_cfg.entity_set_id}'")
            raise UnsupportedModelImplTypeException(entity_set_cfg.entity_set_id, model_cfg.model_id, model_cfg.model_impl)
        
        if model_cfg.model_loading_strategy == "local_disk_storage":
            model_path = self._get_model_path(entity_set_cfg, model_cfg)
            if model_cfg.model_impl == "SequenceTagger":
                model_loader = SequenceTaggerLoader(
                    model_name_or_path=str(model_path), 
                    cache_root=self._frameworks_config_handler.get_flair_cache_root(), 
                    loading_strategy=model_cfg.model_loading_strategy
                )
                model_inference_maker = SequenceTaggerInferenceMaker(model_loader=model_loader)

        elif model_cfg.model_loading_strategy == "huggingface_hub":
            if model_cfg.model_impl == "SequenceTagger":
                model_loader = SequenceTaggerLoader(
                    model_name_or_path=model_cfg.model_name,
                    cache_root=self._frameworks_config_handler.get_flair_cache_root(), 
                    loading_strategy=model_cfg.model_loading_strategy
                )
                model_inference_maker = SequenceTaggerInferenceMaker(model_loader=model_loader)
        
        return model_loader, model_inference_maker
    
    def _check_requirements(self, requirements: List[str]) -> bool:
        """
        Check if all given requirements (like in requirements.txt format)
        are installed in the current environment.
        
        :param requirements: List of requirement strings, e.g. ["flair==0.11.1", "torch>=2.0"]
        :return: True if all requirements are satisfied, False otherwise.
        """
        for req_str in requirements:
            try:
                req = Requirement(req_str)
                installed_version = importlib.metadata.version(req.name)
                if req.specifier and not req.specifier.contains(Version(installed_version), prereleases=True):
                    self.logger.warning(f"{req.name} {installed_version} does not satisfy {req.specifier}")
                    return False
                else:
                    self.logger.info(f"{req.name} {installed_version} satisfies {req.specifier}")
            except importlib.metadata.PackageNotFoundError:
                self.logger.warning(f"{req_str} is not installed")
                return False
            except InvalidVersion:
                self.logger.error(f"Could not parse version for {req_str}")
                return False
        
        return True
    
    def _get_model_path(self, entity_set_cfg, model_cfg) -> Path:
        """
        Construct the path to the model based on the entity set configuration and model configuration.
        
        :param entity_set_cfg: The configuration for the entity set.
        :param model_cfg: The configuration for the model.
        :return: The path to the model.
        """
        return Path(
            *entity_set_cfg.supported_models_root_dir
            ).joinpath(
                *model_cfg.model_directory_name
            ).joinpath(
                model_cfg.model_version
            )
    
    def get_model_inference_maker(self, entity_set_id: str, model_id: str) -> ModelInferenceMaker:
        """
        Retrieve the ModelInferenceMaker for the specified entity set and model ID.
        
        :param entity_set_id: The ID of the entity set for which the model is requested.
        :param model_id: The ID of the model to be used for inference.
        :return: An instance of ModelInferenceMaker for the specified model.
        """
        if entity_set_id not in self._models_registry:
            raise EntitySetNotFoundException(entity_set_id)
        if model_id not in self._models_registry[entity_set_id]:
            raise ModelNotFoundException(entity_set_id, model_id)
        
        _, model_inference_maker = self._models_registry[entity_set_id][model_id]
        return model_inference_maker
    
    def list_models(self, entity_set_id: str) -> Dict[str, str]:
        """
        List available models for a given entity set.
        
        :param entity_set_id: The ID of the entity set for which models are listed.
        :return: A dictionary mapping model IDs to model types.
        """
        return {
            m.model_id: m.model_type
            for m in self._entity_set_models_config_handler.get_entity_set(entity_set_id).supported_models
        }

    def get_entity_set_labels(self, entity_set_id) -> List[str]:
        """
        Get the labels for the entities in the specified entity set.
        
        :param entity_set_id: The ID of the entity set for which labels are requested.
        :return: A list of entity labels.
        """
        entity_set_labels = self._entity_set_models_config_handler.get_entity_set(entity_set_id).entity_set_labels
        labels: List[str] = list()
        for label in entity_set_labels:
            if not label.fine_grained:
                labels.append(label.id)
            for fine_grained_label in label.fine_grained:
                labels.append(fine_grained_label.id)
        return labels

    def get_model_config(self, entity_set_id: str, model_id: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model.
        
        :param entity_set_id: The ID of the entity set for which the model configuration is requested.
        :param model_id: The ID of the model for which the configuration is requested.
        :return: A dictionary containing the model configuration.
        """
        entity_set = self._entity_set_models_config_handler.get_entity_set(entity_set_id)
        if not entity_set:
            raise EntitySetNotFoundException(entity_set_id)
        model_config = next(
            (m for m in entity_set.supported_models if m.model_id == model_id), 
            None
        )
        if not model_config:
            raise ModelNotFoundException(entity_set_id, model_id)
        return model_config.model_dump()

    def reload_model_registry(self) -> None:
        """
        Reload the model registry from the application configuration.
        This method should be called to refresh the model registry, typically after configuration changes.
        
        :return: None
        """
        self._models_registry = self._load_model_registry()