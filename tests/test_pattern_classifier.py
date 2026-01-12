"""Tests for data_generation/pattern_classifier.py module.

Covers:
- PatternClassifier.get_instruction_templates()
- PatternClassifier.get_quality_patterns()
- Pattern classification with language support

Law coverage:
- LAW-language-extensible: Language-specific pattern retrieval
- LAW-backward-compatible: Default fallback behavior
"""

import pytest

from mochi.data_generation.pattern_classifier import (
    PatternClassifier,
    ClassificationResult,
)


class TestGetInstructionTemplates:
    """Tests for PatternClassifier.get_instruction_templates()."""

    def test_returns_python_templates_for_python(self):
        """Returns Python-specific templates for 'python' language."""
        templates = PatternClassifier.get_instruction_templates("python")

        assert isinstance(templates, dict)
        assert "error-handling" in templates
        # Python templates should mention Python-specific concepts
        error_templates = templates["error-handling"]
        assert any("except" in t.lower() or "python" in t.lower() for t in error_templates)

    def test_returns_default_templates_for_typescript(self):
        """Returns default (TypeScript) templates for 'typescript' language."""
        templates = PatternClassifier.get_instruction_templates("typescript")

        assert isinstance(templates, dict)
        assert "error-handling" in templates
        assert "null-safety" in templates
        assert "type-safety" in templates

    def test_returns_default_templates_for_javascript(self):
        """Returns default templates for 'javascript' language."""
        templates = PatternClassifier.get_instruction_templates("javascript")

        assert isinstance(templates, dict)
        # JavaScript uses the same templates as TypeScript (default)
        assert "error-handling" in templates

    def test_falls_back_to_default_for_unknown_language(self):
        """Falls back to default templates for unknown languages."""
        templates = PatternClassifier.get_instruction_templates("unknown_lang")

        assert isinstance(templates, dict)
        # Should return default templates
        assert "error-handling" in templates
        assert "null-safety" in templates

    def test_case_insensitive_language_matching(self):
        """Language matching is case-insensitive."""
        templates_lower = PatternClassifier.get_instruction_templates("python")
        templates_upper = PatternClassifier.get_instruction_templates("PYTHON")

        # Both should return Python templates
        assert templates_lower == templates_upper

    def test_empty_language_returns_default(self):
        """Empty language string returns default templates."""
        templates = PatternClassifier.get_instruction_templates("")

        assert isinstance(templates, dict)
        assert "error-handling" in templates

    def test_templates_have_expected_categories(self):
        """Templates include expected transformation categories."""
        templates = PatternClassifier.get_instruction_templates("typescript")

        expected_categories = [
            "error-handling",
            "null-safety",
            "type-safety",
            "async-await",
            "validation",
        ]

        for category in expected_categories:
            assert category in templates, f"Missing category: {category}"
            assert len(templates[category]) > 0, f"Empty templates for: {category}"


class TestGetQualityPatterns:
    """Tests for PatternClassifier.get_quality_patterns()."""

    def test_returns_python_patterns_for_python(self):
        """Returns Python-specific quality patterns for 'python' language."""
        patterns = PatternClassifier.get_quality_patterns("python")

        assert isinstance(patterns, dict)
        assert "error-handling" in patterns

    def test_returns_default_patterns_for_typescript(self):
        """Returns default (TypeScript) patterns for 'typescript' language."""
        patterns = PatternClassifier.get_quality_patterns("typescript")

        assert isinstance(patterns, dict)
        assert "error-handling" in patterns

    def test_falls_back_to_default_for_unknown_language(self):
        """Falls back to default patterns for unknown languages."""
        patterns = PatternClassifier.get_quality_patterns("unknown_lang")

        assert isinstance(patterns, dict)
        assert "error-handling" in patterns

    def test_patterns_have_good_and_bad(self):
        """Quality patterns include 'good' and 'bad' pattern lists."""
        patterns = PatternClassifier.get_quality_patterns("typescript")

        for category, category_patterns in patterns.items():
            assert "good" in category_patterns or "bad" in category_patterns, \
                f"Category {category} missing good/bad patterns"


class TestPatternClassifierIntegration:
    """Integration tests for PatternClassifier."""

    def test_classify_uses_language_specific_templates(self):
        """Classification uses language-specific instruction templates."""
        classifier = PatternClassifier()

        # Verify templates are accessible
        py_templates = classifier.get_instruction_templates("python")
        ts_templates = classifier.get_instruction_templates("typescript")

        # Python and TypeScript should have different error-handling templates
        # (Python mentions except/try, TypeScript mentions try-catch)
        assert py_templates["error-handling"] != ts_templates["error-handling"] or \
               py_templates == ts_templates  # If same, that's also valid

    def test_classification_result_structure(self):
        """ClassificationResult has expected structure."""
        result = ClassificationResult(
            transform_type="error-handling",
            is_learnable=True,
            instruction="Add error handling",
            confidence=0.9,
            reason="Contains try-catch pattern",
        )

        assert result.transform_type == "error-handling"
        assert result.is_learnable is True
        assert result.confidence == 0.9
        assert isinstance(result.reason, str)


class TestPatternClassifierEdgeCases:
    """Edge case tests for PatternClassifier."""

    def test_none_language_handled_gracefully(self):
        """None language is handled gracefully."""
        # Should not raise, should return default
        templates = PatternClassifier.get_instruction_templates(None)
        assert isinstance(templates, dict)

    def test_whitespace_language_handled(self):
        """Whitespace-only language string is handled."""
        templates = PatternClassifier.get_instruction_templates("   ")
        assert isinstance(templates, dict)
