from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from typing import Any, Dict, List, Optional

import yaml


class SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _replace_placeholders(obj: Any, replacements: Dict[str, str]) -> Any:
    if isinstance(obj, str):
        return obj.format_map(SafeDict(replacements))
    if isinstance(obj, dict):
        return {k: _replace_placeholders(v, replacements) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_placeholders(v, replacements) for v in obj]
    return obj


def _normalize_replacements(replacements: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """
    Convert replacement values to plain strings. Accepts Path objects and other types.

    :param replacements: Original replacements dictionary with values of various types.
    :return: Normalized replacements dictionary with string values.
    """
    if not replacements:
        return dict()
    normalized: Dict[str, str] = dict()
    for k, v in replacements.items():
        if isinstance(v, Path):
            normalized[k] = str(v.resolve())
        else:
            normalized[k] = str(v)
    return normalized


class FlairConfig(BaseModel):
    cache_root_dir: List[str] = Field(..., description="Path components to Flair cache root; may contain placeholders")


class FrameworksConfig(BaseModel):
    flair: Optional[FlairConfig] = None


class FrameworksConfigHandler:
    """
    Loads frameworks_config.yml and optionally replaces placeholders like
    "{PROJECT_ROOT}" by passing replacements to load_from_file / load_from_yaml_string.
    """

    DEFAULT_CONFIG_PATH = (
        Path(__file__).resolve().parent.parent.parent / "configs" / "frameworks_config.yml"
    )

    def __init__(self, raw: Dict[str, Any], config: FrameworksConfig):
        self._raw = raw
        self._config = config

    @classmethod
    def load_from_file(cls, 
                       path: Optional[Path] = None, 
                       replacements: Optional[Dict[str, Any]] = None) -> "FrameworksConfigHandler":
        """
        Load the frameworks configuration from a YAML file.

        :param path: Optional path to the configuration file. If not provided, DEFAULT_CONFIG_PATH is used.
        :param replacements: Optional dictionary of placeholder replacements to apply to the raw YAML content.
                             Path values in this dict will be converted to strings automatically.
        :return: An instance of FrameworksConfigHandler with the loaded configuration.
        """
        cfg_path = Path(path) if path else cls.DEFAULT_CONFIG_PATH
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if replacements:
            normalized = _normalize_replacements(replacements)
            raw = _replace_placeholders(raw, normalized)

        if raw.get("flair") and raw["flair"].get("cache_root_dir") is None:
            raw["flair"]["cache_root_dir"] = list()

        try:
            config = FrameworksConfig(**raw)
        except ValidationError as ve:
            raise ValidationError(ve.errors()) from ve

        return cls(raw=raw, config=config)

    @classmethod
    def load_from_yaml_string(cls, 
                              yaml_str: str, 
                              replacements: Optional[Dict[str, Any]] = None) -> "FrameworksConfigHandler":
        """
        Load the frameworks configuration from a YAML string.

        :param yaml_str: YAML string containing the configuration.
        :param replacements: Optional dictionary of placeholder replacements to apply to the raw YAML content.
                             Path values in this dict will be converted to strings automatically.
        :return: An instance of FrameworksConfigHandler with the loaded configuration.
        """
        raw = yaml.safe_load(yaml_str) or dict()
        if replacements:
            normalized = _normalize_replacements(replacements)
            raw = _replace_placeholders(raw, normalized)

        if raw.get("flair") and raw["flair"].get("cache_root_dir") is None:
            raw["flair"]["cache_root_dir"] = list()

        config = FrameworksConfig(**raw)
        return cls(raw=raw, config=config)

    def get_config(self) -> FrameworksConfig:
        """
        Return the FrameworksConfig instance.
        
        :return: FrameworksConfig instance.
        """
        return self._config

    def get_flair_cache_root(self) -> Optional[Path]:
        """
        Return the flair.cache_root_dir as a Path constructed from the list of path components.
        
        :return: Path to the Flair cache root, or None if not configured.
        """
        flair_cfg = self._config.flair
        if flair_cfg is None or not flair_cfg.cache_root_dir:
            return None
        return Path(*flair_cfg.cache_root_dir)

    def as_dict(self) -> Dict[str, Any]:
        """
        Return the raw configuration as a dictionary.
        
        :return: Dictionary representation of the configuration.
        """
        return self._raw.copy()