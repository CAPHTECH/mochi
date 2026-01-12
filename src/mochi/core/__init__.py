"""Core types and configuration for mochi library."""

from .types import (
    AdapterConfig,
    AdapterType,
    BaseAdapterConfig,
    InferenceConfig,
    ProjectAdapterConfig,
    TrainingConfig,
)
from .config import MochiConfig, load_config
from .exceptions import (
    MochiError,
    AdapterError,
    AdapterNotFoundError,
    TrainingError,
    InferenceError,
    ConfigurationError,
)
from .language_specs import (
    LanguageId,
    LanguageSpec,
    TestFrameworkSpec,
    BlockPattern,
    LANGUAGE_SPECS,
    detect_language,
    is_test_file,
    get_language_spec,
    get_transform_patterns,
    get_test_patterns,
    get_file_patterns_for_languages,
    get_test_file_patterns_for_languages,
    get_supported_languages,
)

__all__ = [
    # Types
    "AdapterConfig",
    "AdapterType",
    "BaseAdapterConfig",
    "ProjectAdapterConfig",
    "TrainingConfig",
    "InferenceConfig",
    # Config
    "MochiConfig",
    "load_config",
    # Exceptions
    "MochiError",
    "AdapterError",
    "AdapterNotFoundError",
    "TrainingError",
    "InferenceError",
    "ConfigurationError",
    # Language specs
    "LanguageId",
    "LanguageSpec",
    "TestFrameworkSpec",
    "BlockPattern",
    "LANGUAGE_SPECS",
    "detect_language",
    "is_test_file",
    "get_language_spec",
    "get_transform_patterns",
    "get_test_patterns",
    "get_file_patterns_for_languages",
    "get_test_file_patterns_for_languages",
    "get_supported_languages",
]
