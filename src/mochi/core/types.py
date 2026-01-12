"""Core type definitions for mochi library.

This module defines the fundamental types used throughout the mochi library,
including adapter configurations, training configurations, and inference settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Union

from .language_specs import (
    LanguageId,
    get_file_patterns_for_languages,
    get_test_file_patterns_for_languages,
)


class AdapterType(str, Enum):
    """Type of adapter."""

    BASE = "base"
    PROJECT = "project"


# Type alias for language specification (LanguageId or string for extensibility)
LanguageSpec = Union[LanguageId, str]


def normalize_language_list(languages: list[LanguageSpec]) -> list[LanguageSpec]:
    """Normalize language list, converting strings to LanguageId where possible.

    This is a shared utility function used by adapter configurations to convert
    string language identifiers to their LanguageId enum values when available.

    Args:
        languages: List of language identifiers (LanguageId or string)

    Returns:
        Normalized list with strings converted to LanguageId where possible
    """
    normalized: list[LanguageSpec] = []
    for lang in languages:
        if isinstance(lang, str):
            try:
                normalized.append(LanguageId(lang))
            except ValueError:
                normalized.append(lang)  # Keep as string for unknown languages
        else:
            normalized.append(lang)
    return normalized


@dataclass
class AdapterConfig:
    """Base configuration for all adapters."""

    name: str
    adapter_type: AdapterType
    base_model: str
    adapter_path: Path | None = None
    description: str = ""
    version: str = "1.0.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.adapter_path, str):
            self.adapter_path = Path(self.adapter_path)


@dataclass
class BaseAdapterConfig(AdapterConfig):
    """Configuration for Base Adapter.

    Base adapters are trained on common patterns across many projects:
    - error-handling
    - async-await
    - type-safety
    - null-safety
    - validation

    Supports multiple languages through LanguageId.
    """

    patterns: list[str] = field(
        default_factory=lambda: [
            "error-handling",
            "async-await",
            "type-safety",
            "null-safety",
            "validation",
        ]
    )
    languages: list[LanguageSpec] = field(default_factory=lambda: ["typescript", "javascript"])

    def __post_init__(self) -> None:
        super().__post_init__()
        self.adapter_type = AdapterType.BASE
        # Convert string languages to LanguageId where possible
        self.languages = normalize_language_list(self.languages)

    @property
    def file_patterns(self) -> list[str]:
        """Get file patterns for configured languages."""
        return get_file_patterns_for_languages(self.languages)


@dataclass
class ProjectAdapterConfig(AdapterConfig):
    """Configuration for Project Adapter.

    Project adapters are trained on project-specific patterns:
    - Custom types and interfaces
    - Project naming conventions
    - Custom error classes
    - Domain-specific utilities

    Supports multiple languages through LanguageId.
    include_patterns is computed from languages by default.
    """

    base_adapter: str | None = None  # Reference to base adapter name
    project_root: Path | None = None
    languages: list[LanguageSpec] = field(default_factory=lambda: ["typescript"])
    _include_patterns: list[str] | None = field(default=None, repr=False)
    exclude_patterns: list[str] = field(
        default_factory=lambda: ["node_modules/**", "*.test.ts", "*.spec.ts", "__pycache__/**"]
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.adapter_type = AdapterType.PROJECT
        if isinstance(self.project_root, str):
            self.project_root = Path(self.project_root)
        # Convert string languages to LanguageId where possible
        self.languages = normalize_language_list(self.languages)

    @property
    def include_patterns(self) -> list[str]:
        """Get include patterns for configured languages.

        If _include_patterns is set explicitly, use that.
        Otherwise, compute from languages.
        """
        if self._include_patterns is not None:
            return self._include_patterns
        return get_file_patterns_for_languages(self.languages)

    @include_patterns.setter
    def include_patterns(self, value: list[str]) -> None:
        """Set explicit include patterns."""
        self._include_patterns = value

    @property
    def test_patterns(self) -> list[str]:
        """Get test file patterns for configured languages."""
        return get_test_file_patterns_for_languages(self.languages)


@dataclass
class TrainingConfig:
    """Configuration for adapter training."""

    # Model settings
    base_model: str = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"

    # LoRA settings
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_layers: int = 16

    # Training hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 4
    epochs: int = 3
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512
    warmup_steps: int = 100

    # Data settings
    train_data_path: Path | None = None
    valid_data_path: Path | None = None
    train_ratio: float = 0.9

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("output/adapter"))
    save_every: int = 100
    eval_every: int = 100

    # Device settings
    use_mps: bool = True  # Use Apple Silicon GPU

    def __post_init__(self) -> None:
        if isinstance(self.train_data_path, str):
            self.train_data_path = Path(self.train_data_path)
        if isinstance(self.valid_data_path, str):
            self.valid_data_path = Path(self.valid_data_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


@dataclass
class InferenceConfig:
    """Configuration for inference."""

    # Generation settings
    max_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.95
    repetition_penalty: float = 1.1

    # Adapter stacking weights
    base_weight: float = 0.3
    project_weight: float = 0.7

    # LSP integration
    use_lsp_context: bool = True
    lsp_context_lines: int = 50

    # Caching
    cache_enabled: bool = True
    cache_max_size: int = 1000


@dataclass
class AdapterStackConfig:
    """Configuration for stacking multiple adapters."""

    adapters: list[tuple[str, float]] = field(default_factory=list)
    # List of (adapter_name, weight) tuples

    def validate(self) -> bool:
        """Validate that weights sum to 1.0."""
        if not self.adapters:
            return False
        total_weight = sum(weight for _, weight in self.adapters)
        return abs(total_weight - 1.0) < 0.001

    def normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0."""
        if not self.adapters:
            return
        total_weight = sum(weight for _, weight in self.adapters)
        if total_weight > 0:
            self.adapters = [
                (name, weight / total_weight) for name, weight in self.adapters
            ]
