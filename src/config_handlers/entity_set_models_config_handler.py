from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from typing import Any, Dict, List, Optional

import yaml


class SupportedModel(BaseModel):
    model_id: str
    model_name: str
    model_description: str = ""
    model_links: List[str] = Field(default_factory=list)
    model_type: Optional[str] = None
    model_type_description: Optional[str] = None
    model_loading_strategy: Optional[str] = None
    model_directory_name: List[str] = Field(default_factory=list)
    model_version: Optional[str] = None
    model_impl: Optional[str] = None
    model_system_requirements: List[str] = Field(default_factory=list)
    model_tokenizer_params: Dict[str, Any] = Field(default_factory=dict)
    model_generate_params: Dict[str, Any] = Field(default_factory=dict)


class FineGrainedLabel(BaseModel):
    id: str
    description: str = ""
    background_color: Optional[str] = None
    border_color: Optional[str] = None


class EntitySetLabel(BaseModel):
    id: str
    description: str = ""
    background_color: Optional[str] = None
    border_color: Optional[str] = None
    fine_grained: List[FineGrainedLabel] = Field(default_factory=list)


class EntitySetModel(BaseModel):
    entity_set_id: str
    corpus_name: str
    corpus_doctypes: List[str] = Field(default_factory=list)
    corpus_description: Optional[str] = None
    corpus_version: Optional[str] = None
    corpus_languages: List[str] = Field(default_factory=list)
    corpus_links: List[str] = Field(default_factory=list)
    entity_set_labels: List[EntitySetLabel] = Field(default_factory=list)
    sample_texts: List[str] = Field(default_factory=list)
    supported_models_root_dir: List[str] = Field(default_factory=list)
    supported_models: List[SupportedModel] = Field(default_factory=list)


class EntitySetModelsConfigHandler:
    """
    Loads and provides access to the entity set models configuration.
    """

    DEFAULT_CONFIG_PATH = (
        Path(__file__).resolve().parent.parent.parent / "configs" / "entity_set_models_config.yml"
    )

    def __init__(self, entity_sets: List[EntitySetModel]):
        self._entity_sets: List[EntitySetModel] = entity_sets
        self._by_id: Dict[str, EntitySetModel] = {es.entity_set_id: es for es in entity_sets}

    @classmethod
    def load_from_file(cls, path: Optional[Path] = None) -> "EntitySetModelsConfigHandler":
        """
        Load configuration from YAML file.

        :param path: Optional path to the configuration file. If not provided, DEFAULT_CONFIG_PATH is used.
        :return: An instance of EntitySetModelsConfigHandler with the loaded configuration.
        :raises FileNotFoundError: If the configuration file does not exist.
        :raises ValidationError: If the YAML structure is invalid.
        """
        cfg_path = Path(path) if path else cls.DEFAULT_CONFIG_PATH
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or dict()
        raw_entities = raw.get("entity_set_models", list())

        entity_sets: List[EntitySetModel] = list()
        errors: List[str] = list()
        for idx, es in enumerate(raw_entities):
            try:
                entity_sets.append(EntitySetModel(**es))
            except ValidationError as ve:
                errors.append(f"Entity set index {idx} validation error: {ve}")

        if errors:
            raise ValidationError(errors)

        return cls(entity_sets)

    @classmethod
    def load_from_yaml_string(cls, yaml_str: str) -> "EntitySetModelsConfigHandler":
        """
        Load configuration from a YAML string.
        
        :param yaml_str: YAML string containing the configuration.
        :return: An instance of EntitySetModelsConfigHandler with the loaded configuration.
        """
        raw = yaml.safe_load(yaml_str) or dict()
        raw_entities = raw.get("entity_set_models", list())
        entity_sets = [EntitySetModel(**es) for es in raw_entities]
        return cls(entity_sets)

    @property
    def entity_sets(self) -> List[EntitySetModel]:
        """
        Get a list of all entity sets.

        :return: List of all entity sets.
        """
        return list(self._entity_sets)

    @property
    def entity_set_ids(self) -> List[str]:
        """
        Get a list of all entity set IDs.

        :return: List of all entity set IDs.
        """
        return [es.entity_set_id for es in self._entity_sets]

    def get_entity_set(self, entity_set_id: str) -> Optional[EntitySetModel]:
        """
        Return the EntitySetModel for the given id or None if not found.

        :param entity_set_id: The id of the entity set.
        :return: EntitySetModel object or None.
        """
        return self._by_id.get(entity_set_id)
    
    def get_supported_model_ids_for_entity_set(self, entity_set_id: str) -> List[str]:
        """
        Return the list of supported model ids for the given entity set id.
        If the entity set is not found, returns an empty list.

        :param entity_set_id: The id of the entity set.
        :return: List of supported model ids.
        """
        es = self.get_entity_set(entity_set_id)
        if es:
            return [model.model_id for model in es.supported_models]
        return list()

    def get_supported_models_for_entity_set(self, entity_set_id: str) -> List[SupportedModel]:
        """
        Return the list of SupportedModel for the given entity set id.
        If the entity set is not found, returns an empty list.

        :param entity_set_id: The id of the entity set.
        :return: List of SupportedModel objects.
        """
        es = self.get_entity_set(entity_set_id)
        return es.supported_models if es else list()

    def get_supported_model(self, entity_set_id: str, model_id: str) -> Optional[SupportedModel]:
        """
        Return the SupportedModel for the given entity set id and model id.
        If not found, returns None.

        :param entity_set_id: The id of the entity set.
        :param model_id: The id of the model.
        :return: SupportedModel object or None.
        """
        models = self.get_supported_models_for_entity_set(entity_set_id)
        for m in models:
            if m.model_id == model_id:
                return m
        return None