"""Training data generation modules."""

from .alpaca_converter import AlpacaConverter
from .diff_extractor import (
    CodeTransformPair,
    GitDiffExtractor,
    TRANSFORM_PATTERNS,
    extract_transforms_from_repos,
)
from .pattern_classifier import ClassificationResult, PatternClassifier
from .test_patterns import (
    TEST_INSTRUCTION_TEMPLATES,
    TEST_QUALITY_PATTERNS,
    TEST_TRANSFORM_PATTERNS,
    TestExample,
    TestPatternGenerator,
)

__all__ = [
    # alpaca_converter
    "AlpacaConverter",
    # diff_extractor
    "CodeTransformPair",
    "GitDiffExtractor",
    "TRANSFORM_PATTERNS",
    "extract_transforms_from_repos",
    # pattern_classifier
    "ClassificationResult",
    "PatternClassifier",
    # test_patterns
    "TEST_INSTRUCTION_TEMPLATES",
    "TEST_QUALITY_PATTERNS",
    "TEST_TRANSFORM_PATTERNS",
    "TestExample",
    "TestPatternGenerator",
]
