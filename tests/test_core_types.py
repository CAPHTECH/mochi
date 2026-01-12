"""Tests for core/types.py module.

Covers:
- normalize_language_list() function
- BaseAdapterConfig language normalization
- ProjectAdapterConfig language normalization

Law coverage:
- LAW-backward-compatible: Existing behavior preserved
- LAW-language-extensible: LanguageSpec-based extensibility
"""

import pytest
from pathlib import Path

from mochi.core.types import (
    normalize_language_list,
    BaseAdapterConfig,
    ProjectAdapterConfig,
    AdapterType,
)
from mochi.core.language_specs import LanguageId


class TestNormalizeLanguageList:
    """Tests for normalize_language_list function."""

    def test_converts_known_strings_to_language_id(self):
        """Known language strings are converted to LanguageId enum."""
        result = normalize_language_list(["typescript", "python", "javascript"])

        assert result == [
            LanguageId.TYPESCRIPT,
            LanguageId.PYTHON,
            LanguageId.JAVASCRIPT,
        ]

    def test_preserves_unknown_languages_as_strings(self):
        """Unknown language strings are preserved as-is."""
        result = normalize_language_list(["typescript", "unknown_lang", "custom"])

        assert result[0] == LanguageId.TYPESCRIPT
        assert result[1] == "unknown_lang"
        assert result[2] == "custom"

    def test_empty_list_returns_empty(self):
        """Empty input returns empty list."""
        result = normalize_language_list([])
        assert result == []

    def test_preserves_existing_language_id(self):
        """LanguageId values pass through unchanged."""
        input_list = [LanguageId.PYTHON, "typescript", LanguageId.GO]
        result = normalize_language_list(input_list)

        assert result == [
            LanguageId.PYTHON,
            LanguageId.TYPESCRIPT,
            LanguageId.GO,
        ]

    def test_case_sensitive_matching(self):
        """Language matching is case-sensitive (lowercase required)."""
        # LanguageId values are lowercase
        result = normalize_language_list(["Python", "TYPESCRIPT"])

        # These should remain as strings since they don't match enum values
        assert result[0] == "Python"
        assert result[1] == "TYPESCRIPT"


class TestBaseAdapterConfig:
    """Tests for BaseAdapterConfig language normalization."""

    def test_normalizes_string_languages_on_init(self):
        """String languages are normalized to LanguageId on initialization."""
        config = BaseAdapterConfig(
            name="test-adapter",
            adapter_type=AdapterType.BASE,
            base_model="test-model",
            languages=["python", "typescript"],
        )

        assert config.languages == [LanguageId.PYTHON, LanguageId.TYPESCRIPT]

    def test_default_languages_are_normalized(self):
        """Default languages (typescript, javascript) are normalized."""
        config = BaseAdapterConfig(
            name="test-adapter",
            adapter_type=AdapterType.BASE,
            base_model="test-model",
        )

        assert LanguageId.TYPESCRIPT in config.languages
        assert LanguageId.JAVASCRIPT in config.languages

    def test_preserves_unknown_languages(self):
        """Unknown languages are preserved as strings."""
        config = BaseAdapterConfig(
            name="test-adapter",
            adapter_type=AdapterType.BASE,
            base_model="test-model",
            languages=["python", "custom_lang"],
        )

        assert config.languages[0] == LanguageId.PYTHON
        assert config.languages[1] == "custom_lang"

    def test_file_patterns_property(self):
        """file_patterns returns patterns for configured languages."""
        config = BaseAdapterConfig(
            name="test-adapter",
            adapter_type=AdapterType.BASE,
            base_model="test-model",
            languages=["python"],
        )

        patterns = config.file_patterns
        assert "*.py" in patterns


class TestProjectAdapterConfig:
    """Tests for ProjectAdapterConfig language normalization."""

    def test_normalizes_string_languages_on_init(self):
        """String languages are normalized to LanguageId on initialization."""
        config = ProjectAdapterConfig(
            name="test-project",
            adapter_type=AdapterType.PROJECT,
            base_model="test-model",
            languages=["go", "rust"],
        )

        assert config.languages == [LanguageId.GO, LanguageId.RUST]

    def test_default_language_is_typescript(self):
        """Default language is typescript (normalized to LanguageId)."""
        config = ProjectAdapterConfig(
            name="test-project",
            adapter_type=AdapterType.PROJECT,
            base_model="test-model",
        )

        assert config.languages == [LanguageId.TYPESCRIPT]

    def test_include_patterns_from_languages(self):
        """include_patterns is computed from languages."""
        config = ProjectAdapterConfig(
            name="test-project",
            adapter_type=AdapterType.PROJECT,
            base_model="test-model",
            languages=["python"],
        )

        patterns = config.include_patterns
        assert "*.py" in patterns

    def test_test_patterns_from_languages(self):
        """test_patterns returns test file patterns for languages."""
        config = ProjectAdapterConfig(
            name="test-project",
            adapter_type=AdapterType.PROJECT,
            base_model="test-model",
            languages=["python"],
        )

        patterns = config.test_patterns
        assert "test_*.py" in patterns or "*_test.py" in patterns

    def test_project_root_path_conversion(self):
        """project_root string is converted to Path."""
        config = ProjectAdapterConfig(
            name="test-project",
            adapter_type=AdapterType.PROJECT,
            base_model="test-model",
            project_root="/some/path",
        )

        assert isinstance(config.project_root, Path)
        assert config.project_root == Path("/some/path")
