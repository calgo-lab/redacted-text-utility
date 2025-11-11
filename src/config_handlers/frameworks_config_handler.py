from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError


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


class FlairConfig(BaseModel):
    cache_root: str = Field(..., description="Path to Flair cache root; may contain placeholders")


class FrameworksConfig(BaseModel):
    flair: Optional[FlairConfig] = None


class FrameworksConfigHandler:
    """
    Loads frameworks_config.yml and optionally replaces placeholders like
    "{project_root}" by passing replacements to load_from_file / load_from_yaml_string.
    """

    DEFAULT_CONFIG_PATH = (
        Path(__file__).resolve().parent.parent.parent / "configs/frameworks_config.yml"
    )

    def __init__(self, raw: Dict[str, Any], config: FrameworksConfig):
        self._raw = raw
        self._config = config

    @classmethod
    def load_from_file(cls, 
                       path: Optional[Path] = None, 
                       replacements: Optional[Dict[str, str]] = None) -> "FrameworksConfigHandler":

        """
        Load the frameworks configuration from a YAML file.
        
        :param path: Optional path to the configuration file. If not provided, DEFAULT_CONFIG_PATH is used.
        :param replacements: Optional dictionary of placeholder replacements to apply to the raw YAML content.
        :return: An instance of FrameworksConfigHandler with the loaded configuration.
        """
        
        cfg_path = Path(path) if path else cls.DEFAULT_CONFIG_PATH
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or dict()
        if replacements:
            raw = _replace_placeholders(raw, replacements)

        try:
            config = FrameworksConfig(**raw)
        except ValidationError as ve:
            raise ValidationError(ve.errors()) from ve

        return cls(raw=raw, config=config)

    @classmethod
    def load_from_yaml_string(cls, 
                              yaml_str: str, 
                              replacements: Optional[Dict[str, str]] = None) -> "FrameworksConfigHandler":
        
        """
        Load the frameworks configuration from a YAML string.
        
        :param yaml_str: YAML string containing the configuration.
        :param replacements: Optional dictionary of placeholder replacements to apply to the raw YAML content.
        :return: An instance of FrameworksConfigHandler with the loaded configuration.
        """
        raw = yaml.safe_load(yaml_str) or {}
        if replacements:
            raw = _replace_placeholders(raw, replacements)
        config = FrameworksConfig(**raw)
        return cls(raw=raw, config=config)

    def get_config(self) -> FrameworksConfig:
        return self._config

    def get_flair_cache_root(self, replacements: Optional[Dict[str, str]] = None) -> Optional[Path]:
        """
        Return the flair.cache_root as a Path. If replacements are provided they
        are applied to the original raw value.

        :param replacements: Optional dictionary of placeholder replacements.
        :return: Path to the Flair cache root, or None if not configured.
        """
        flair_cfg = self._config.flair
        if flair_cfg is None:
            return None

        cache_root_raw = flair_cfg.cache_root
        if replacements:
            raw_flair = (self._raw.get("flair") or dict())
            cache_root_raw = raw_flair.get("cache_root", cache_root_raw)
            cache_root_raw = _replace_placeholders(cache_root_raw, replacements)

        return Path(cache_root_raw).expanduser()

    def as_dict(self) -> Dict[str, Any]:
        return self._raw.copy()