"""Language specifications for multi-language support.

This module defines language-specific configurations that enable mochi to support
multiple programming languages. Each language has its own patterns for:
- File extensions and patterns
- Test frameworks (vitest, pytest, etc.)
- Code transformation patterns (error-handling, type-safety, etc.)
- Block detection (braces vs indentation)
- LSP server commands

Laws (ELD):
- L-language-extensible: New language support requires only adding to LANGUAGE_SPECS
- L-backward-compatible: Existing TypeScript behavior remains unchanged
- L-language-detection: File path uniquely determines language

Terms (ELD):
- LanguageSpec: Central configuration for a programming language
- LanguageId: Unique identifier for a language
- TestFrameworkSpec: Test framework specific patterns
- BlockPattern: Code block detection pattern
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class LanguageId(str, Enum):
    """Supported programming languages.

    Use string enum for easy serialization and backward compatibility.
    """
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    PYTHON = "python"
    RUST = "rust"
    GO = "go"


@dataclass(frozen=True)
class TestFrameworkSpec:
    """Test framework specification.

    Defines patterns for detecting and classifying test code for a specific
    test framework (e.g., vitest, pytest, unittest).

    Attributes:
        name: Framework name (e.g., "vitest", "pytest")
        file_patterns: Glob patterns for test files (e.g., "*.spec.ts", "test_*.py")
        structure_patterns: Regex patterns for test structure (describe, def test_, etc.)
        assertion_patterns: Regex patterns for assertions (expect, assert, etc.)
        mock_patterns: Regex patterns for mocking
        setup_patterns: Regex patterns for setup/teardown
    """
    name: str
    file_patterns: tuple[str, ...]
    structure_patterns: tuple[str, ...]
    assertion_patterns: tuple[str, ...]
    mock_patterns: tuple[str, ...] = ()
    setup_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class BlockPattern:
    """Pattern for detecting code blocks (functions, classes, etc.).

    Different languages use different block delimiters:
    - TypeScript/JavaScript: braces {}
    - Python: indentation

    Attributes:
        name: Pattern name (e.g., "function", "class")
        start_pattern: Regex to detect block start
        uses_braces: Whether the language uses braces for blocks
        uses_indentation: Whether the language uses indentation for blocks
    """
    name: str
    start_pattern: str
    uses_braces: bool = True
    uses_indentation: bool = False


@dataclass(frozen=True)
class LanguageSpec:
    """Complete specification for a programming language.

    This is the central configuration object for language-specific behavior.
    All language-dependent logic should reference this spec.

    Attributes:
        id: Unique language identifier
        display_name: Human-readable name
        file_extensions: File extensions including the dot (e.g., ".ts", ".py")
        file_patterns: Glob patterns for source files (e.g., "*.ts", "*.py")
        test_frameworks: Supported test frameworks
        transform_patterns: Patterns for code transformation classification
        block_patterns: Patterns for detecting code blocks
        lsp_server_command: Command to start LSP server
        comment_style: Comment syntax ("//", "#", etc.)
    """
    id: LanguageId
    display_name: str
    file_extensions: tuple[str, ...]
    file_patterns: tuple[str, ...]
    test_frameworks: tuple[TestFrameworkSpec, ...]
    transform_patterns: dict[str, tuple[str, ...]]
    block_patterns: tuple[BlockPattern, ...]
    lsp_server_command: tuple[str, ...]
    comment_style: str = "//"

    def matches_file(self, file_path: str | Path) -> bool:
        """Check if a file path matches this language's extensions.

        Args:
            file_path: Path to check

        Returns:
            True if the file extension matches this language
        """
        path_str = str(file_path).lower()
        return any(path_str.endswith(ext) for ext in self.file_extensions)

    def is_test_file(self, file_path: str | Path) -> bool:
        """Check if a file path is a test file for this language.

        Args:
            file_path: Path to check

        Returns:
            True if the file matches any test file pattern
        """
        path_str = str(file_path)
        filename = Path(path_str).name

        for framework in self.test_frameworks:
            for pattern in framework.file_patterns:
                # Check both full path and filename
                if fnmatch.fnmatch(filename, pattern):
                    return True
                if fnmatch.fnmatch(path_str, pattern):
                    return True
                # Handle ** patterns
                if "**" in pattern and fnmatch.fnmatch(path_str, f"*/{pattern}"):
                    return True
        return False

    def get_test_file_patterns(self) -> list[str]:
        """Get all test file patterns for this language.

        Returns:
            List of glob patterns for test files
        """
        patterns: list[str] = []
        for framework in self.test_frameworks:
            patterns.extend(framework.file_patterns)
        return patterns

    def get_all_transform_patterns(self) -> dict[str, tuple[str, ...]]:
        """Get all transform patterns including test patterns.

        Returns:
            Dictionary of transform type to regex patterns
        """
        all_patterns = dict(self.transform_patterns)

        # Add test patterns from frameworks
        test_patterns: dict[str, list[str]] = {
            "test-structure": [],
            "test-assertion": [],
            "test-mock": [],
            "test-setup": [],
        }

        for framework in self.test_frameworks:
            test_patterns["test-structure"].extend(framework.structure_patterns)
            test_patterns["test-assertion"].extend(framework.assertion_patterns)
            test_patterns["test-mock"].extend(framework.mock_patterns)
            test_patterns["test-setup"].extend(framework.setup_patterns)

        # Add non-empty test patterns
        for key, patterns in test_patterns.items():
            if patterns:
                all_patterns[key] = tuple(patterns)

        return all_patterns


# =============================================================================
# TypeScript/JavaScript Specification
# =============================================================================

_VITEST_JEST_FRAMEWORK = TestFrameworkSpec(
    name="vitest",  # Also compatible with Jest
    file_patterns=(
        "*.spec.ts", "*.test.ts",
        "*.spec.tsx", "*.test.tsx",
        "*.spec.js", "*.test.js",
        "*.spec.jsx", "*.test.jsx",
    ),
    structure_patterns=(
        r"\+\s*describe\s*\(",
        r"\+\s*it\s*\(",
        r"\+\s*test\s*\(",
        r"\+\s*it\.each\s*\(",
        r"\+\s*describe\.each\s*\(",
    ),
    assertion_patterns=(
        r"\+\s*expect\s*\(",
        r"\+\s*\.toBe\s*\(",
        r"\+\s*\.toEqual\s*\(",
        r"\+\s*\.toContain\s*\(",
        r"\+\s*\.toThrow\s*\(",
        r"\+\s*\.toHaveBeenCalled",
        r"\+\s*\.toBeNull\s*\(",
        r"\+\s*\.toBeDefined\s*\(",
        r"\+\s*\.toHaveLength\s*\(",
    ),
    mock_patterns=(
        r"\+\s*vi\.mock\s*\(",
        r"\+\s*jest\.mock\s*\(",
        r"\+\s*vi\.fn\s*\(",
        r"\+\s*jest\.fn\s*\(",
        r"\+\s*vi\.spyOn\s*\(",
        r"\+\s*jest\.spyOn\s*\(",
        r"\+\s*mockImplementation\s*\(",
        r"\+\s*mockReturnValue\s*\(",
        r"\+\s*mockResolvedValue\s*\(",
    ),
    setup_patterns=(
        r"\+\s*beforeEach\s*\(",
        r"\+\s*afterEach\s*\(",
        r"\+\s*beforeAll\s*\(",
        r"\+\s*afterAll\s*\(",
    ),
)

_TYPESCRIPT_TRANSFORM_PATTERNS: dict[str, tuple[str, ...]] = {
    "error-handling": (
        r"\+.*try\s*\{",
        r"\+.*catch\s*\(",
        r"\+.*throw\s+new",
        r"\+.*Result<",
        r"\+.*\.catch\s*\(",
        r"\+.*finally\s*\{",
    ),
    "null-safety": (
        r"\+.*\?\.",
        r"\+.*\?\?",
        r"\+.*!= null",
        r"\+.*!== null",
        r"\+.*!= undefined",
        r"\+.*!== undefined",
    ),
    "type-safety": (
        r"\+.*:\s*[A-Z][a-zA-Z0-9_]+\s*[=;]",
        r"\+.*as\s+[A-Z][a-zA-Z0-9_]+",
        r"\+.*is\s+[A-Z][a-zA-Z0-9_]+",
        r"\+.*<[A-Z][a-zA-Z0-9_]+>",
    ),
    "async-await": (
        r"\+.*async\s+function",
        r"\+.*async\s*\(",
        r"\+.*await\s+",
    ),
    "validation": (
        r"\+.*\.parse\s*\(",
        r"\+.*\.safeParse\s*\(",
        r"\+.*validate\w*\s*\(",
        r"\+.*z\.\w+\s*\(",
        r"\+.*assert\w*\s*\(",
    ),
}

_TYPESCRIPT_BLOCK_PATTERNS = (
    BlockPattern(
        name="function",
        start_pattern=r"^\s*(?:export\s+)?(?:async\s+)?function\s+\w+",
        uses_braces=True,
    ),
    BlockPattern(
        name="arrow_function",
        start_pattern=r"^\s*(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\(",
        uses_braces=True,
    ),
    BlockPattern(
        name="class",
        start_pattern=r"^\s*(?:export\s+)?class\s+\w+",
        uses_braces=True,
    ),
    BlockPattern(
        name="object",
        start_pattern=r"^\s*(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*\{",
        uses_braces=True,
    ),
    BlockPattern(
        name="method",
        start_pattern=r"^\s+(?:async\s+)?\w+\s*\([^)]*\)\s*(?::\s*\w+)?\s*\{",
        uses_braces=True,
    ),
)

TYPESCRIPT_SPEC = LanguageSpec(
    id=LanguageId.TYPESCRIPT,
    display_name="TypeScript",
    file_extensions=(".ts", ".tsx"),
    file_patterns=("*.ts", "*.tsx"),
    test_frameworks=(_VITEST_JEST_FRAMEWORK,),
    transform_patterns=_TYPESCRIPT_TRANSFORM_PATTERNS,
    block_patterns=_TYPESCRIPT_BLOCK_PATTERNS,
    lsp_server_command=("npx", "typescript-language-server", "--stdio"),
    comment_style="//",
)

JAVASCRIPT_SPEC = LanguageSpec(
    id=LanguageId.JAVASCRIPT,
    display_name="JavaScript",
    file_extensions=(".js", ".jsx", ".mjs", ".cjs"),
    file_patterns=("*.js", "*.jsx", "*.mjs", "*.cjs"),
    test_frameworks=(_VITEST_JEST_FRAMEWORK,),
    transform_patterns=_TYPESCRIPT_TRANSFORM_PATTERNS,  # Same patterns work
    block_patterns=_TYPESCRIPT_BLOCK_PATTERNS,
    lsp_server_command=("npx", "typescript-language-server", "--stdio"),
    comment_style="//",
)


# =============================================================================
# Python Specification
# =============================================================================

_PYTEST_FRAMEWORK = TestFrameworkSpec(
    name="pytest",
    file_patterns=(
        "test_*.py",
        "*_test.py",
        "tests/**/*.py",
    ),
    structure_patterns=(
        r"\+\s*def\s+test_\w+",
        r"\+\s*class\s+Test\w+",
        r"\+\s*@pytest\.mark\.\w+",
        r"\+\s*@pytest\.fixture",
    ),
    assertion_patterns=(
        r"\+\s*assert\s+",
        r"\+\s*pytest\.raises\s*\(",
        r"\+\s*pytest\.approx\s*\(",
        r"\+\s*pytest\.warns\s*\(",
    ),
    mock_patterns=(
        r"\+\s*@patch\s*\(",
        r"\+\s*@patch\.object\s*\(",
        r"\+\s*MagicMock\s*\(",
        r"\+\s*Mock\s*\(",
        r"\+\s*AsyncMock\s*\(",
        r"\+\s*mocker\.\w+",
        r"\+\s*monkeypatch\.\w+",
    ),
    setup_patterns=(
        r"\+\s*@pytest\.fixture",
        r"\+\s*def\s+setup_method\s*\(",
        r"\+\s*def\s+teardown_method\s*\(",
        r"\+\s*def\s+setup_class\s*\(",
        r"\+\s*def\s+teardown_class\s*\(",
        r"\+\s*def\s+setup_module\s*\(",
        r"\+\s*def\s+teardown_module\s*\(",
    ),
)

_UNITTEST_FRAMEWORK = TestFrameworkSpec(
    name="unittest",
    file_patterns=(
        "test_*.py",
        "*_test.py",
    ),
    structure_patterns=(
        r"\+\s*class\s+Test\w+\s*\(\s*unittest\.TestCase\s*\)",
        r"\+\s*def\s+test_\w+\s*\(\s*self\s*\)",
    ),
    assertion_patterns=(
        r"\+\s*self\.assert\w+\s*\(",
        r"\+\s*self\.assertEqual\s*\(",
        r"\+\s*self\.assertTrue\s*\(",
        r"\+\s*self\.assertFalse\s*\(",
        r"\+\s*self\.assertRaises\s*\(",
        r"\+\s*self\.assertIn\s*\(",
    ),
    mock_patterns=(
        r"\+\s*@patch\s*\(",
        r"\+\s*@patch\.object\s*\(",
        r"\+\s*MagicMock\s*\(",
        r"\+\s*Mock\s*\(",
    ),
    setup_patterns=(
        r"\+\s*def\s+setUp\s*\(\s*self\s*\)",
        r"\+\s*def\s+tearDown\s*\(\s*self\s*\)",
        r"\+\s*def\s+setUpClass\s*\(",
        r"\+\s*def\s+tearDownClass\s*\(",
    ),
)

_PYTHON_TRANSFORM_PATTERNS: dict[str, tuple[str, ...]] = {
    "error-handling": (
        r"\+\s*try\s*:",
        r"\+\s*except\s+\w*",
        r"\+\s*raise\s+\w+",
        r"\+\s*finally\s*:",
        r"\+.*Result\[",  # For Result type pattern
    ),
    "null-safety": (
        r"\+.*is\s+None",
        r"\+.*is\s+not\s+None",
        r"\+.*if\s+\w+\s*:",
        r"\+.*Optional\[",
        r"\+.*\|\s*None",  # Python 3.10+ union syntax
    ),
    "type-safety": (
        r"\+.*:\s*[A-Z][a-zA-Z0-9_]+",
        r"\+.*->\s*[A-Z][a-zA-Z0-9_]+",
        r"\+.*:\s*list\[",
        r"\+.*:\s*dict\[",
        r"\+.*:\s*tuple\[",
        r"\+.*@dataclass",
        r"\+.*TypeVar\s*\(",
    ),
    "async-await": (
        r"\+\s*async\s+def",
        r"\+\s*await\s+",
        r"\+.*asyncio\.\w+",
        r"\+.*aiohttp\.\w+",
    ),
    "validation": (
        r"\+.*pydantic\.\w+",
        r"\+.*@validator\s*\(",
        r"\+.*@field_validator\s*\(",
        r"\+.*BaseModel",
        r"\+.*Field\s*\(",
    ),
}

_PYTHON_BLOCK_PATTERNS = (
    BlockPattern(
        name="function",
        start_pattern=r"^\s*(?:async\s+)?def\s+\w+",
        uses_braces=False,
        uses_indentation=True,
    ),
    BlockPattern(
        name="class",
        start_pattern=r"^\s*class\s+\w+",
        uses_braces=False,
        uses_indentation=True,
    ),
    BlockPattern(
        name="method",
        start_pattern=r"^\s+(?:async\s+)?def\s+\w+",
        uses_braces=False,
        uses_indentation=True,
    ),
)

PYTHON_SPEC = LanguageSpec(
    id=LanguageId.PYTHON,
    display_name="Python",
    file_extensions=(".py", ".pyi"),
    file_patterns=("*.py", "*.pyi"),
    test_frameworks=(_PYTEST_FRAMEWORK, _UNITTEST_FRAMEWORK),
    transform_patterns=_PYTHON_TRANSFORM_PATTERNS,
    block_patterns=_PYTHON_BLOCK_PATTERNS,
    lsp_server_command=("pylsp",),
    comment_style="#",
)


# =============================================================================
# Language Registry
# =============================================================================

LANGUAGE_SPECS: dict[LanguageId, LanguageSpec] = {
    LanguageId.TYPESCRIPT: TYPESCRIPT_SPEC,
    LanguageId.JAVASCRIPT: JAVASCRIPT_SPEC,
    LanguageId.PYTHON: PYTHON_SPEC,
}

# Extension to language mapping for quick lookup
_EXTENSION_TO_LANGUAGE: dict[str, LanguageId] = {}
for _spec in LANGUAGE_SPECS.values():
    for _ext in _spec.file_extensions:
        _EXTENSION_TO_LANGUAGE[_ext] = _spec.id


# =============================================================================
# Utility Functions
# =============================================================================

def get_language_spec(language_id: LanguageId | str) -> LanguageSpec | None:
    """Get language specification by ID.

    Args:
        language_id: Language identifier (string or enum)

    Returns:
        LanguageSpec if found, None otherwise
    """
    if isinstance(language_id, str):
        try:
            language_id = LanguageId(language_id)
        except ValueError:
            return None
    return LANGUAGE_SPECS.get(language_id)


def detect_language(file_path: str | Path) -> LanguageId | None:
    """Detect language from file path.

    Law compliance:
    - L-language-detection: File path uniquely determines language

    Args:
        file_path: Path to the file

    Returns:
        LanguageId if detected, None otherwise
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    return _EXTENSION_TO_LANGUAGE.get(ext)


def is_test_file(file_path: str | Path) -> bool:
    """Check if file is a test file for any supported language.

    Uses language-specific patterns:
    - TypeScript: *.spec.ts, *.test.ts
    - Python: test_*.py, *_test.py

    Args:
        file_path: Path to the file

    Returns:
        True if the file is a test file
    """
    language_id = detect_language(file_path)
    if language_id is None:
        return False

    spec = LANGUAGE_SPECS.get(language_id)
    if spec is None:
        return False

    return spec.is_test_file(file_path)


def get_transform_patterns(language_id: LanguageId | str) -> dict[str, tuple[str, ...]]:
    """Get transformation patterns for a language.

    Args:
        language_id: Language identifier

    Returns:
        Dictionary of transform type to regex patterns
    """
    spec = get_language_spec(language_id)
    if spec is None:
        return {}
    return spec.transform_patterns


def get_test_patterns(language_id: LanguageId | str) -> dict[str, tuple[str, ...]]:
    """Get test patterns for a language (combined from all frameworks).

    Args:
        language_id: Language identifier

    Returns:
        Dictionary of test type to regex patterns
    """
    spec = get_language_spec(language_id)
    if spec is None:
        return {}

    patterns: dict[str, list[str]] = {
        "test-structure": [],
        "test-assertion": [],
        "test-mock": [],
        "test-setup": [],
    }

    for framework in spec.test_frameworks:
        patterns["test-structure"].extend(framework.structure_patterns)
        patterns["test-assertion"].extend(framework.assertion_patterns)
        patterns["test-mock"].extend(framework.mock_patterns)
        patterns["test-setup"].extend(framework.setup_patterns)

    return {k: tuple(v) for k, v in patterns.items() if v}


def get_file_patterns_for_languages(
    language_ids: list[LanguageId | str] | None = None,
) -> list[str]:
    """Get file patterns for specified languages.

    Args:
        language_ids: List of language identifiers. If None, returns patterns for all languages.

    Returns:
        List of glob patterns
    """
    if language_ids is None:
        language_ids = list(LANGUAGE_SPECS.keys())

    patterns: list[str] = []
    for lang_id in language_ids:
        spec = get_language_spec(lang_id)
        if spec:
            patterns.extend(spec.file_patterns)

    return patterns


def get_test_file_patterns_for_languages(
    language_ids: list[LanguageId | str] | None = None,
) -> list[str]:
    """Get test file patterns for specified languages.

    Args:
        language_ids: List of language identifiers. If None, returns patterns for all languages.

    Returns:
        List of glob patterns for test files
    """
    if language_ids is None:
        language_ids = list(LANGUAGE_SPECS.keys())

    patterns: list[str] = []
    for lang_id in language_ids:
        spec = get_language_spec(lang_id)
        if spec:
            patterns.extend(spec.get_test_file_patterns())

    return patterns


def get_supported_languages() -> list[LanguageId]:
    """Get list of supported languages.

    Returns:
        List of supported LanguageIds
    """
    return list(LANGUAGE_SPECS.keys())
