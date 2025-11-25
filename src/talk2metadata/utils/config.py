"""Configuration management for Talk2Metadata."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .logging import get_logger

logger = get_logger(__name__)


class Config:
    """Configuration manager for Talk2Metadata."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration.

        Args:
            config_dict: Configuration dictionary. If None, uses defaults.
        """
        self._config = config_dict or self._get_default_config()

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "run_id": None,  # Optional run ID for organizing multiple runs
            "data": {
                "raw_dir": "./data/raw",
                "processed_dir": "./data/processed",
                "indexes_dir": "./data/indexes",
                "metadata_dir": "./data/metadata",
            },
            "schema": {
                "fk_detection": {
                    "use_heuristics": True,
                    "use_agent": True,  # Enable agent-based FK detection
                    "agent_trigger": "auto",  # auto | always | never
                    "agent_threshold": 2,  # Use agent if < N FKs found by rules
                    "inclusion_tolerance": 0.1,  # 10% mismatch allowed
                    "min_coverage": 0.9,  # 90% coverage required for FK
                    "min_overlap_ratio": 0.8,  # 80% value overlap for candidates
                },
            },
            "ingest": {
                "target_table": None,
                "data_type": None,
                "source_path": None,
            },
            "agent": {
                "enabled": False,  # Currently disabled
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.0,
            },
            "evaluation": {
                "top_k": 10,  # Number of results to retrieve per query
                "output_format": "text",  # Display format: "text" or "json"
                "save_format": "both",  # Save format: "json", "txt", or "both"
                "auto_save": True,  # Auto-save results to benchmark directory
                "evaluate_all_modes": False,  # Evaluate all enabled modes by default
            },
            "qa_generation": {
                "num_patterns": 15,
                "instances_per_pattern": 5,
                "validate": True,
                "filter_valid": True,
                "auto_save": True,
            },
            "modes": {
                # Mode-specific configurations
                "record_embedding": {
                    "indexer": {
                        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                        "device": None,
                        "batch_size": 32,
                        "normalize": True,
                    },
                    "retriever": {
                        "top_k": 5,
                        "similarity_metric": "cosine",
                        "per_table_top_k": 5,
                        "use_reranking": False,
                    },
                },
                # Global mode settings
                "active": "record_embedding",  # Active mode name
                "compare": {  # Comparison mode settings
                    "enabled": False,  # Enable comparison mode
                    "modes": [],  # List of modes to compare (empty = all enabled)
                },
            },
        }

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> Config:
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Config instance

        Example:
            >>> config = Config.from_yaml("config.yml")
            >>> print(config.get("data.raw_dir"))
        """
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        logger.info(f"Loading config from {yaml_path}")

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Merge with defaults
        default_config = cls._get_default_config()
        merged_config = cls._merge_configs(default_config, config_dict or {})

        return cls(merged_config)

    @staticmethod
    def _merge_configs(
        base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recursively merge two configuration dictionaries.

        Args:
            base: Base configuration
            override: Override configuration

        Returns:
            Merged configuration
        """
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = Config._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Supports dot notation for nested keys.

        Args:
            key: Configuration key (e.g., "data.raw_dir")
            default: Default value if key not found

        Returns:
            Configuration value

        Example:
            >>> config.get("data.raw_dir")
            './data/raw'
            >>> config.get("schema.fk_detection.min_coverage")
            0.9
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key.

        Supports dot notation for nested keys.

        Args:
            key: Configuration key (e.g., "data.raw_dir")
            value: Value to set
        """
        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration dictionary
        """
        return self._config.copy()

    def save(self, yaml_path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            yaml_path: Path to save YAML file
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving config to {yaml_path}")

        with open(yaml_path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        """String representation."""
        return f"Config({self._config})"


# Global config instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance.

    If not set, tries to load from config.yml in current directory,
    otherwise uses defaults.

    Returns:
        Global Config instance
    """
    global _global_config
    if _global_config is None:
        # Priority: TALK2METADATA_CONFIG env var > ./config.yml > defaults
        env_path = os.getenv("TALK2METADATA_CONFIG")
        if env_path:
            path = Path(env_path)
            if path.exists():
                try:
                    logger.info(f"Loading config from TALK2METADATA_CONFIG: {path}")
                    _global_config = Config.from_yaml(path)
                    return _global_config
                except Exception as e:
                    logger.warning(
                        f"Failed to load config from TALK2METADATA_CONFIG ({path}): {e}; falling back"
                    )
            else:
                logger.warning(
                    f"TALK2METADATA_CONFIG set to {path} but file does not exist; falling back"
                )

        # Try local config.yml
        config_path = Path("config.yml")
        if config_path.exists():
            try:
                _global_config = Config.from_yaml(config_path)
            except Exception as e:
                logger.warning(f"Failed to load config.yml: {e}, using defaults")
                _global_config = Config()
        else:
            _global_config = Config()
    return _global_config


def set_config(config: Config) -> None:
    """Set global configuration instance.

    Args:
        config: Config instance to set as global
    """
    global _global_config
    _global_config = config


def load_config(yaml_path: str | Path) -> Config:
    """Load configuration from YAML and set as global.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        Loaded Config instance
    """
    config = Config.from_yaml(yaml_path)
    set_config(config)
    return config
