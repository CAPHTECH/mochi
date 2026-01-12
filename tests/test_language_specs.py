"""Tests for core/language_specs.py module.

Covers:
- LanguageId enum
- detect_language() function
- get_language_spec() function
- get_file_patterns_for_languages() function
- get_test_file_patterns_for_languages() function

Law coverage:
- LAW-language-extensible: LanguageSpec-based extensibility
"""

import pytest

from mochi.core.language_specs import (
    LanguageId,
    LanguageSpec,
    detect_language,
    get_language_spec,
    get_file_patterns_for_languages,
    get_test_file_patterns_for_languages,
)


class TestLanguageId:
    """Tests for LanguageId enum."""

    def test_supported_languages_exist(self):
        """Supported programming languages are defined."""
        assert LanguageId.PYTHON.value == "python"
        assert LanguageId.TYPESCRIPT.value == "typescript"
        assert LanguageId.JAVASCRIPT.value == "javascript"
        assert LanguageId.GO.value == "go"
        assert LanguageId.RUST.value == "rust"

    def test_language_id_from_string(self):
        """LanguageId can be created from string value."""
        assert LanguageId("python") == LanguageId.PYTHON
        assert LanguageId("typescript") == LanguageId.TYPESCRIPT

    def test_invalid_language_raises_value_error(self):
        """Invalid language string raises ValueError."""
        with pytest.raises(ValueError):
            LanguageId("nonexistent_language")


class TestDetectLanguage:
    """Tests for detect_language function."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("main.py", LanguageId.PYTHON),
            ("app.ts", LanguageId.TYPESCRIPT),
            ("index.js", LanguageId.JAVASCRIPT),
        ],
    )
    def test_detects_language_from_extension(self, filename, expected):
        """Language is detected from file extension."""
        result = detect_language(filename)
        assert result == expected

    def test_returns_none_for_unknown_extension(self):
        """Returns None for unknown file extensions."""
        result = detect_language("file.xyz")
        assert result is None

    def test_returns_none_for_unsupported_language(self):
        """Returns None for languages without specs (Go, Rust)."""
        # Go and Rust are defined in LanguageId but have no specs
        assert detect_language("main.go") is None
        assert detect_language("lib.rs") is None

    def test_handles_path_with_directories(self):
        """Handles full paths with directories."""
        result = detect_language("/path/to/project/src/main.py")
        assert result == LanguageId.PYTHON

    def test_tsx_detected_as_typescript(self):
        """TSX files are detected as TypeScript."""
        result = detect_language("Component.tsx")
        assert result == LanguageId.TYPESCRIPT

    def test_jsx_detected_as_javascript(self):
        """JSX files are detected as JavaScript."""
        result = detect_language("Component.jsx")
        assert result == LanguageId.JAVASCRIPT


class TestGetLanguageSpec:
    """Tests for get_language_spec function."""

    def test_returns_spec_for_known_language(self):
        """Returns LanguageSpec for known languages."""
        spec = get_language_spec(LanguageId.PYTHON)

        assert isinstance(spec, LanguageSpec)
        assert spec.id == LanguageId.PYTHON
        assert "*.py" in spec.file_patterns

    def test_returns_spec_for_typescript(self):
        """Returns correct spec for TypeScript."""
        spec = get_language_spec(LanguageId.TYPESCRIPT)

        assert spec.id == LanguageId.TYPESCRIPT
        assert "*.ts" in spec.file_patterns
        assert "*.tsx" in spec.file_patterns

    def test_returns_none_for_unsupported_language(self):
        """Returns None for languages without specs."""
        spec = get_language_spec(LanguageId.GO)
        assert spec is None

    def test_spec_has_transform_patterns(self):
        """LanguageSpec includes transform_patterns."""
        spec = get_language_spec(LanguageId.PYTHON)

        assert hasattr(spec, "transform_patterns")
        assert isinstance(spec.transform_patterns, dict)

    def test_spec_has_test_frameworks(self):
        """LanguageSpec includes test frameworks with file patterns."""
        spec = get_language_spec(LanguageId.PYTHON)

        assert hasattr(spec, "test_frameworks")
        assert len(spec.test_frameworks) > 0
        # Test frameworks have file_patterns
        pytest_framework = spec.test_frameworks[0]
        assert hasattr(pytest_framework, "file_patterns")


class TestGetFilePatternsForLanguages:
    """Tests for get_file_patterns_for_languages function."""

    def test_returns_patterns_for_single_language(self):
        """Returns file patterns for a single language."""
        patterns = get_file_patterns_for_languages([LanguageId.PYTHON])

        assert "*.py" in patterns

    def test_returns_combined_patterns_for_multiple_languages(self):
        """Returns combined patterns for multiple languages."""
        patterns = get_file_patterns_for_languages([
            LanguageId.PYTHON,
            LanguageId.TYPESCRIPT,
        ])

        assert "*.py" in patterns
        assert "*.ts" in patterns

    def test_handles_language_id_enum(self):
        """Handles LanguageId enum values."""
        patterns = get_file_patterns_for_languages([
            LanguageId.PYTHON,
            LanguageId.TYPESCRIPT,
        ])

        assert "*.py" in patterns
        assert "*.ts" in patterns

    def test_empty_list_returns_empty(self):
        """Empty language list returns empty patterns."""
        patterns = get_file_patterns_for_languages([])
        assert patterns == []


class TestGetTestFilePatternsForLanguages:
    """Tests for get_test_file_patterns_for_languages function."""

    def test_returns_test_patterns_for_python(self):
        """Returns Python test file patterns."""
        patterns = get_test_file_patterns_for_languages([LanguageId.PYTHON])

        # Python uses test_*.py or *_test.py convention
        assert any("test" in p for p in patterns)

    def test_returns_test_patterns_for_typescript(self):
        """Returns TypeScript test file patterns."""
        patterns = get_test_file_patterns_for_languages([LanguageId.TYPESCRIPT])

        # TypeScript uses *.test.ts or *.spec.ts convention
        assert any("test" in p or "spec" in p for p in patterns)

    def test_combines_patterns_for_multiple_languages(self):
        """Combines test patterns for multiple languages."""
        patterns = get_test_file_patterns_for_languages([
            LanguageId.PYTHON,
            LanguageId.TYPESCRIPT,
        ])

        # Should have both Python and TypeScript patterns
        assert len(patterns) > 0
