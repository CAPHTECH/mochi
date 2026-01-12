"""Training data generation modules.

Note: TRANSFORM_PATTERNS and TEST_TRANSFORM_PATTERNS have been removed.
Use language_specs.LanguageSpec for language-specific patterns instead.
"""

from .alpaca_converter import AlpacaConverter
from .diff_extractor import (
    CodeTransformPair,
    GitDiffExtractor,
    extract_transforms_from_repos,
)
from .pattern_classifier import ClassificationResult, PatternClassifier
from .test_patterns import (
    TEST_INSTRUCTION_TEMPLATES,
    TEST_QUALITY_PATTERNS,
    TestExample,
    TestPatternGenerator,
)

__all__ = [
    # alpaca_converter
    "AlpacaConverter",
    # diff_extractor
    "CodeTransformPair",
    "GitDiffExtractor",
    "extract_transforms_from_repos",
    # pattern_classifier
    "ClassificationResult",
    "PatternClassifier",
    # test_patterns
    "TEST_INSTRUCTION_TEMPLATES",
    "TEST_QUALITY_PATTERNS",
    "TestExample",
    "TestPatternGenerator",
]
