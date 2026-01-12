"""Configuration management for mochi library."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigurationError
from .language_specs import LanguageId
from .types import (
    BaseAdapterConfig,
    InferenceConfig,
    ProjectAdapterConfig,
    TrainingConfig,
)


def _serialize_languages(languages: list) -> list[str]:
    """Serialize languages list to strings for YAML/JSON output."""
    result = []
    for lang in languages:
        if isinstance(lang, LanguageId):
            result.append(lang.value)
        else:
            result.append(str(lang))
    return result


@dataclass
class MochiConfig:
    """Main configuration for mochi library.

    This configuration can be loaded from:
    - mochi.yaml in the project root
    - Environment variables (MOCHI_*)
    - Programmatic configuration
    """

    # Paths
    adapters_dir: Path = field(default_factory=lambda: Path("adapters"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    data_dir: Path = field(default_factory=lambda: Path("data"))

    # Default base model
    base_model: str = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"

    # Registered adapters
    base_adapters: dict[str, BaseAdapterConfig] = field(default_factory=dict)
    project_adapters: dict[str, ProjectAdapterConfig] = field(default_factory=dict)

    # Default training config
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Default inference config
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def __post_init__(self) -> None:
        if isinstance(self.adapters_dir, str):
            self.adapters_dir = Path(self.adapters_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)

    def get_adapter_path(self, adapter_name: str) -> Path:
        """Get the path to an adapter by name."""
        return self.adapters_dir / adapter_name

    def register_base_adapter(self, config: BaseAdapterConfig) -> None:
        """Register a base adapter configuration."""
        self.base_adapters[config.name] = config

    def register_project_adapter(self, config: ProjectAdapterConfig) -> None:
        """Register a project adapter configuration."""
        self.project_adapters[config.name] = config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "adapters_dir": str(self.adapters_dir),
            "output_dir": str(self.output_dir),
            "data_dir": str(self.data_dir),
            "base_model": self.base_model,
            "base_adapters": {
                name: {
                    "name": cfg.name,
                    "adapter_path": str(cfg.adapter_path) if cfg.adapter_path else None,
                    "patterns": cfg.patterns,
                    "languages": _serialize_languages(cfg.languages),
                }
                for name, cfg in self.base_adapters.items()
            },
            "project_adapters": {
                name: {
                    "name": cfg.name,
                    "adapter_path": str(cfg.adapter_path) if cfg.adapter_path else None,
                    "base_adapter": cfg.base_adapter,
                    "project_root": str(cfg.project_root) if cfg.project_root else None,
                    "languages": _serialize_languages(cfg.languages),
                }
                for name, cfg in self.project_adapters.items()
            },
        }

    def save(self, path: Path | str) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MochiConfig:
        """Create configuration from dictionary."""
        config = cls(
            adapters_dir=Path(data.get("adapters_dir", "adapters")),
            output_dir=Path(data.get("output_dir", "output")),
            data_dir=Path(data.get("data_dir", "data")),
            base_model=data.get("base_model", "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"),
        )

        # Load base adapters
        for name, adapter_data in data.get("base_adapters", {}).items():
            config.register_base_adapter(
                BaseAdapterConfig(
                    name=adapter_data.get("name", name),
                    adapter_type=adapter_data.get("adapter_type", "base"),
                    base_model=config.base_model,
                    adapter_path=Path(adapter_data["adapter_path"]) if adapter_data.get("adapter_path") else None,
                    patterns=adapter_data.get("patterns", []),
                    languages=adapter_data.get("languages", ["typescript"]),
                )
            )

        # Load project adapters
        for name, adapter_data in data.get("project_adapters", {}).items():
            config.register_project_adapter(
                ProjectAdapterConfig(
                    name=adapter_data.get("name", name),
                    adapter_type=adapter_data.get("adapter_type", "project"),
                    base_model=config.base_model,
                    adapter_path=Path(adapter_data["adapter_path"]) if adapter_data.get("adapter_path") else None,
                    base_adapter=adapter_data.get("base_adapter"),
                    project_root=Path(adapter_data["project_root"]) if adapter_data.get("project_root") else None,
                    languages=adapter_data.get("languages", ["typescript"]),
                )
            )

        return config


def load_config(
    config_path: Path | str | None = None,
    search_paths: list[Path | str] | None = None,
) -> MochiConfig:
    """Load mochi configuration.

    Search order:
    1. Explicit config_path if provided
    2. MOCHI_CONFIG environment variable
    3. search_paths if provided
    4. Default locations: ./mochi.yaml, ~/.mochi/config.yaml

    Args:
        config_path: Explicit path to configuration file
        search_paths: Additional paths to search for configuration

    Returns:
        MochiConfig instance

    Raises:
        ConfigurationError: If configuration file has errors
    """
    # Build search order
    paths_to_check: list[Path] = []

    if config_path:
        paths_to_check.append(Path(config_path))

    if env_config := os.environ.get("MOCHI_CONFIG"):
        paths_to_check.append(Path(env_config))

    if search_paths:
        paths_to_check.extend(Path(p) for p in search_paths)

    # Default locations
    paths_to_check.extend([
        Path.cwd() / "mochi.yaml",
        Path.cwd() / "mochi.yml",
        Path.home() / ".mochi" / "config.yaml",
    ])

    # Try to load from each path
    for path in paths_to_check:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    if path.suffix in (".yaml", ".yml"):
                        data = yaml.safe_load(f)
                    else:
                        data = json.load(f)
                return MochiConfig.from_dict(data or {})
            except (yaml.YAMLError, json.JSONDecodeError) as e:
                raise ConfigurationError(
                    f"Failed to parse configuration file: {path}",
                    {"path": str(path), "error": str(e)},
                ) from e

    # Return default configuration if no file found
    return MochiConfig()


def get_default_config() -> MochiConfig:
    """Get default mochi configuration."""
    return MochiConfig()
