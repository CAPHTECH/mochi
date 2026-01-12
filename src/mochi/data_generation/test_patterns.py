"""Generate training data for test pattern generation.

This module specializes in extracting and formatting test-related patterns
(describe/it/expect, mocks, fixtures) for training code generation models.

Complements the general-purpose diff_extractor.py with test-specific logic.
Supports multiple languages through the language_specs module.
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from mochi.core.language_specs import (
    LanguageId,
    LanguageSpec,
    LANGUAGE_SPECS,
    detect_language,
    get_test_file_patterns_for_languages,
    get_test_patterns,
    get_language_spec,
)
from .diff_extractor import CodeTransformPair, GitDiffExtractor

logger = logging.getLogger(__name__)


# Legacy test-specific transform patterns (kept for backward compatibility)
# New code should use get_test_patterns() from language_specs
_LEGACY_TEST_TRANSFORM_PATTERNS: dict[str, list[str]] = {
    "test-structure": [
        r"\+\s*describe\s*\(",
        r"\+\s*it\s*\(",
        r"\+\s*test\s*\(",
        r"\+\s*it\.each\s*\(",
        r"\+\s*describe\.each\s*\(",
        # Python pytest
        r"\+\s*def\s+test_\w+",
        r"\+\s*class\s+Test\w+",
    ],
    "test-assertion": [
        r"\+\s*expect\s*\(",
        r"\+\s*assert\s*\(",
        r"\+\s*assert\w+\s*\(",
        r"\+\s*\.toBe\s*\(",
        r"\+\s*\.toEqual\s*\(",
        r"\+\s*\.toContain\s*\(",
        r"\+\s*\.toThrow\s*\(",
        r"\+\s*\.toHaveBeenCalled",
        # Python pytest
        r"\+\s*assert\s+",
        r"\+\s*pytest\.raises",
    ],
    "test-setup": [
        r"\+\s*beforeEach\s*\(",
        r"\+\s*afterEach\s*\(",
        r"\+\s*beforeAll\s*\(",
        r"\+\s*afterAll\s*\(",
        # Python pytest
        r"\+\s*@pytest\.fixture",
        r"\+\s*def\s+setup_",
        r"\+\s*def\s+teardown_",
    ],
    "test-mock": [
        r"\+\s*vi\.mock\s*\(",
        r"\+\s*jest\.mock\s*\(",
        r"\+\s*vi\.fn\s*\(",
        r"\+\s*jest\.fn\s*\(",
        r"\+\s*vi\.spyOn\s*\(",
        r"\+\s*jest\.spyOn\s*\(",
        r"\+\s*mockImplementation\s*\(",
        r"\+\s*mockReturnValue\s*\(",
        # Python pytest
        r"\+\s*@patch",
        r"\+\s*MagicMock",
        r"\+\s*mocker\.",
    ],
}

# Backward compatibility alias
TEST_TRANSFORM_PATTERNS = _LEGACY_TEST_TRANSFORM_PATTERNS

# Instruction templates for test generation
TEST_INSTRUCTION_TEMPLATES: dict[str, list[str]] = {
    "test-structure": [
        "Write a unit test for this function:",
        "Create a test case for this implementation:",
        "Complete the test describe/it structure:",
        "Write the test body with proper assertions:",
        "Add a test case for this behavior:",
        "Create a test block for this functionality:",
    ],
    "test-assertion": [
        "Add assertions to verify the behavior:",
        "Write expect statements to validate:",
        "Add proper assertions for this test:",
        "Complete the expect statements:",
        "Verify the expected behavior with assertions:",
        "Add assertions to check the return value:",
    ],
    "test-setup": [
        "Set up test fixtures:",
        "Write the setup/teardown for this test:",
        "Initialize test dependencies in beforeEach:",
        "Add proper test setup and cleanup:",
        "Configure the test environment:",
    ],
    "test-mock": [
        "Set up mocks for this dependency:",
        "Create mock implementations:",
        "Mock the external dependencies:",
        "Write mock setup for this test:",
        "Replace the dependency with a mock:",
        "Spy on the function calls:",
    ],
}

# Quality patterns for test code (includes TypeScript/JavaScript and Python)
TEST_QUALITY_PATTERNS: dict[str, dict[str, list[str]]] = {
    "test-structure": {
        "good": [
            r"describe\s*\(\s*['\"].*['\"]\s*,",  # Descriptive test name
            r"it\s*\(\s*['\"]should\s+",  # BDD style naming
            r"test\s*\(\s*['\"].*['\"]\s*,",  # Test with description
            # Python pytest
            r"def\s+test_\w+.*:",  # Proper test function
            r"class\s+Test\w+:",  # Test class
            r"@pytest\.mark\.",  # Proper pytest markers
        ],
        "bad": [
            r"it\s*\(\s*['\"]['\"]",  # Empty test description
            r"\.only\s*\(",  # Test exclusivity (it.only, describe.only)
            r"\.skip\s*\(",  # Skipped tests
            # Python pytest
            r"@pytest\.mark\.skip",  # Skipped tests
            r"def\s+test_\s*\(",  # Empty test name
        ],
    },
    "test-assertion": {
        "good": [
            r"expect\s*\([^)]+\)\s*\.\w+\s*\(",  # Complete assertion
            r"toHaveBeenCalledWith\s*\(",  # Specific call verification
            r"toThrowError\s*\(['\"]",  # Error message check
            # Python pytest
            r"assert\s+\w+\s*==",  # Equality assertion
            r"assert\s+\w+\s+in\s+",  # Membership assertion
            r"pytest\.raises\s*\(\w+",  # Exception assertion
        ],
        "bad": [
            r"expect\s*\(\s*\)\s*\.",  # Empty expect
            r"expect\s*\(true\)\s*\.toBe\s*\(true\)",  # Trivial assertion
            # Python pytest
            r"assert\s+True",  # Trivial assertion
            r"assert\s+1\s*==\s*1",  # Trivial assertion
        ],
    },
    "test-setup": {
        "good": [
            r"beforeEach\s*\(\s*(?:async\s*)?\(\s*\)",  # Proper setup hook
            r"afterEach\s*\(\s*(?:async\s*)?\(\s*\)",  # Proper teardown
            # Python pytest
            r"@pytest\.fixture",  # Proper fixture
            r"def\s+\w+\s*\(\s*\w+\s*\)",  # Fixture with scope
        ],
        "bad": [
            r"beforeEach\s*\(\s*\(\s*\)\s*=>\s*\{\s*\}\s*\)",  # Empty setup
            # Python pytest
            r"@pytest\.fixture\s*\(\s*\)\s*\ndef\s+\w+.*:\s*pass",  # Empty fixture
        ],
    },
    "test-mock": {
        "good": [
            r"mockImplementation\s*\(",  # Custom mock implementation
            r"mockResolvedValue\s*\(",  # Async mock
            r"mockReturnValue\s*\(",  # Sync mock
            # Python pytest
            r"@patch\s*\(['\"][\w\.]+['\"]",  # Proper patch target
            r"MagicMock\s*\(.*spec=",  # Mock with spec
            r"mocker\.patch",  # pytest-mock
        ],
        "bad": [
            r"vi\.mock\s*\(['\"]['\"]",  # Empty mock path
            # Python pytest
            r"@patch\s*\(\s*\)",  # Empty patch
        ],
    },
}


@dataclass
class TestExample:
    """Training example for test pattern generation."""

    instruction: str
    input_code: str
    output_code: str
    transform_type: str
    confidence: float
    file_path: str = ""
    language: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class TestPatternGenerator:
    """Generate training examples for test patterns.

    Extracts test-related code transformations from git history and
    formats them as training examples for code generation models.
    Supports multiple languages through the language_specs module.
    """

    def __init__(
        self,
        repo_path: Path | None = None,
        file_patterns: list[str] | None = None,
        languages: list[LanguageId | str] | None = None,
    ):
        """Initialize generator.

        Args:
            repo_path: Path to git repository
            file_patterns: Glob patterns for test files (overrides languages)
            languages: List of languages to process (default: typescript)
        """
        self.repo_path = repo_path
        self.languages = languages

        # Determine file patterns
        if file_patterns:
            # Explicit file_patterns takes precedence
            self.file_patterns = file_patterns
        elif languages:
            # Generate from languages
            self.file_patterns = get_test_file_patterns_for_languages(languages)
        else:
            # Default: TypeScript only (backward compatibility)
            self.file_patterns = ["*.spec.ts", "*.test.ts", "*.spec.tsx", "*.test.tsx"]

    def extract_from_git(
        self,
        max_commits: int = 500,
        transform_types: list[str] | None = None,
    ) -> list[CodeTransformPair]:
        """Extract test transformation pairs from git history.

        Args:
            max_commits: Maximum commits to analyze
            transform_types: Test transform types to extract

        Returns:
            List of CodeTransformPair for test patterns
        """
        if not self.repo_path:
            raise ValueError("repo_path is required for git extraction")

        extractor = _TestDiffExtractor(self.repo_path, languages=self.languages)
        pairs = extractor.extract_transforms(
            file_patterns=self.file_patterns,
            max_commits=max_commits,
            transform_types=transform_types or list(TEST_TRANSFORM_PATTERNS.keys()),
        )

        return pairs

    def generate_examples(
        self,
        pairs: list[CodeTransformPair],
        min_confidence: float = 0.3,
    ) -> list[TestExample]:
        """Generate training examples from transformation pairs.

        Args:
            pairs: Extracted transformation pairs
            min_confidence: Minimum confidence threshold

        Returns:
            List of TestExample instances
        """
        examples: list[TestExample] = []

        for pair in pairs:
            example = self._create_example(pair)
            if example and example.confidence >= min_confidence:
                examples.append(example)

        logger.info(f"Generated {len(examples)} test examples from {len(pairs)} pairs")
        return examples

    def _create_example(self, pair: CodeTransformPair) -> TestExample | None:
        """Create a training example from a transformation pair."""
        transform_type = pair.transform_type

        # Classify and validate
        confidence = self._calculate_confidence(pair)
        if confidence < 0.1:
            return None

        # Select instruction
        templates = TEST_INSTRUCTION_TEMPLATES.get(transform_type, ["Write test code:"])
        instruction = random.choice(templates)

        # Get language from pair (if available)
        language = getattr(pair, "language", "") or ""

        return TestExample(
            instruction=instruction,
            input_code=pair.before_code,
            output_code=pair.after_code,
            transform_type=transform_type,
            confidence=confidence,
            file_path=pair.file_path,
            language=language,
            metadata={
                "commit_hash": pair.commit_hash,
                "commit_message": pair.commit_message,
                "language": language,
            },
        )

    def _calculate_confidence(self, pair: CodeTransformPair) -> float:
        """Calculate confidence score for a transformation pair."""
        transform_type = pair.transform_type
        after_code = pair.after_code

        good_patterns = TEST_QUALITY_PATTERNS.get(transform_type, {}).get("good", [])
        bad_patterns = TEST_QUALITY_PATTERNS.get(transform_type, {}).get("bad", [])

        good_matches = sum(1 for p in good_patterns if re.search(p, after_code, re.MULTILINE))
        bad_matches = sum(1 for p in bad_patterns if re.search(p, after_code, re.MULTILINE))

        if good_patterns:
            base_score = good_matches / len(good_patterns)
        else:
            base_score = 0.5

        # Penalize bad patterns
        penalty = bad_matches * 0.2
        confidence = max(0.0, min(1.0, base_score - penalty))

        return confidence

    def generate_from_test_files(
        self,
        test_dir: Path,
        target_src_dir: Path | None = None,
    ) -> list[TestExample]:
        """Generate examples by pairing source files with their tests.

        Args:
            test_dir: Directory containing test files
            target_src_dir: Directory containing source files (for pairing)

        Returns:
            List of TestExample for test generation training
        """
        examples: list[TestExample] = []

        # Build test file patterns from configured file_patterns
        for pattern in self.file_patterns:
            for test_file in test_dir.rglob(pattern):
                example = self._create_example_from_file(test_file, target_src_dir)
                if example:
                    examples.append(example)

        logger.info(f"Generated {len(examples)} examples from test files")
        return examples

    def _create_example_from_file(
        self,
        test_file: Path,
        src_dir: Path | None,
    ) -> TestExample | None:
        """Create example from a test file."""
        try:
            test_content = test_file.read_text()
        except Exception as e:
            logger.debug(f"Failed to read {test_file}: {e}")
            return None

        # Detect language
        try:
            language_id = detect_language(str(test_file))
            language = language_id.value if language_id else ""
        except Exception:
            language = ""

        # Extract test blocks
        test_blocks = self._extract_test_blocks(test_content, language)
        if not test_blocks:
            return None

        # Find corresponding source file
        source_context = ""
        if src_dir:
            source_file = self._find_source_file(test_file, src_dir, language)
            if source_file and source_file.exists():
                try:
                    source_context = source_file.read_text()[:2000]  # Limit context
                except Exception:
                    pass

        # Pick a random test block for training
        test_block = random.choice(test_blocks)

        instruction = random.choice([
            "Write a unit test for this function:",
            "Create a test case for this implementation:",
            "Write test assertions for this code:",
        ])

        # Build context with appropriate comment style
        if source_context:
            comment_prefix = "#" if language == "python" else "//"
            input_code = f"{comment_prefix} Source file context:\n{source_context}"
        else:
            input_code = test_block["context"]

        return TestExample(
            instruction=instruction,
            input_code=input_code,
            output_code=test_block["code"],
            transform_type="test-structure",
            confidence=0.7,
            file_path=str(test_file),
            language=language,
        )

    def _extract_test_blocks(self, content: str, language: str = "") -> list[dict[str, str]]:
        """Extract individual test blocks from test file content.

        Args:
            content: File content
            language: Language identifier (e.g., "python", "typescript")

        Returns:
            List of dicts with "code" and "context" keys
        """
        blocks: list[dict[str, str]] = []

        if language == "python":
            # Python: Extract def test_* functions
            test_pattern = re.compile(
                r'^(\s*)(def\s+test_\w+\s*\([^)]*\)\s*(?:->\s*\w+\s*)?:)',
                re.MULTILINE,
            )

            lines = content.split('\n')
            for match in test_pattern.finditer(content):
                start = match.start()
                start_line = content[:start].count('\n')
                indent_str = match.group(1)
                base_indent = len(indent_str)

                # Find end of function by indentation
                end_line = start_line + 1
                while end_line < len(lines):
                    line = lines[end_line]
                    stripped = line.strip()
                    # Skip empty lines and comments
                    if not stripped or stripped.startswith('#'):
                        end_line += 1
                        continue
                    # Check if we've exited the function
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= base_indent and stripped:
                        break
                    end_line += 1

                block_code = '\n'.join(lines[start_line:end_line])
                # Get context (imports and class wrapper)
                context_start = max(0, start_line - 20)
                context = '\n'.join(lines[context_start:start_line])

                blocks.append({
                    "code": block_code,
                    "context": context,
                })
        else:
            # TypeScript/JavaScript: Extract it/test blocks
            it_pattern = re.compile(
                r'((?:it|test)\s*\(\s*[\'"].*?[\'"]\s*,\s*(?:async\s*)?\([^)]*\)\s*=>\s*\{)',
                re.MULTILINE | re.DOTALL,
            )

            for match in it_pattern.finditer(content):
                start = match.start()
                # Find matching closing brace
                brace_count = 1
                end = match.end()

                while end < len(content) and brace_count > 0:
                    if content[end] == "{":
                        brace_count += 1
                    elif content[end] == "}":
                        brace_count -= 1
                    end += 1

                if brace_count == 0:
                    block_code = content[start:end]
                    # Get context (imports and describe wrapper)
                    context_start = max(0, start - 500)
                    context = content[context_start:start]

                    blocks.append({
                        "code": block_code,
                        "context": context,
                    })

        return blocks

    def _find_source_file(
        self,
        test_file: Path,
        src_dir: Path,
        language: str = "",
    ) -> Path | None:
        """Find source file corresponding to a test file.

        Args:
            test_file: Path to test file
            src_dir: Directory to search for source file
            language: Language identifier for test file patterns

        Returns:
            Path to source file if found
        """
        test_name = test_file.stem

        if language == "python":
            # Python patterns:
            # test_foo.py -> foo.py
            # foo_test.py -> foo.py
            # conftest.py -> (no source)
            if test_name == "conftest":
                return None

            if test_name.startswith("test_"):
                source_name = test_name[5:]  # Remove "test_" prefix
            elif test_name.endswith("_test"):
                source_name = test_name[:-5]  # Remove "_test" suffix
            else:
                source_name = test_name

            # Search for Python source file
            for ext in [".py"]:
                source_path = src_dir / f"{source_name}{ext}"
                if source_path.exists():
                    return source_path

                # Try nested directories
                for match in src_dir.rglob(f"{source_name}{ext}"):
                    return match
        else:
            # TypeScript/JavaScript patterns:
            # foo.spec.ts -> foo.ts
            # foo.test.ts -> foo.ts
            for suffix in [".spec", ".test"]:
                if test_name.endswith(suffix):
                    source_name = test_name[: -len(suffix)]
                    break
            else:
                source_name = test_name

            # Search in src directory
            for ext in [".ts", ".tsx", ".js", ".jsx"]:
                source_path = src_dir / f"{source_name}{ext}"
                if source_path.exists():
                    return source_path

                # Try nested directories
                for match in src_dir.rglob(f"{source_name}{ext}"):
                    return match

        return None

    def to_jsonl(self, examples: list[TestExample], output_path: Path) -> None:
        """Write examples to JSONL file in Alpaca format.

        Args:
            examples: List of TestExample
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for example in examples:
                record = {
                    "instruction": example.instruction,
                    "input": f"// File: {example.file_path}\n{example.input_code}",
                    "output": example.output_code,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Wrote {len(examples)} examples to {output_path}")

    def to_mlx_format(self, examples: list[TestExample], output_path: Path) -> None:
        """Write examples in mlx-lm training format.

        Args:
            examples: List of TestExample
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for example in examples:
                text = (
                    f"### Instruction:\n{example.instruction}\n\n"
                    f"### Input:\n// File: {example.file_path}\n{example.input_code}\n\n"
                    f"### Response:\n{example.output_code}"
                )
                record = {"text": text}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Wrote {len(examples)} examples to {output_path}")


class _TestDiffExtractor(GitDiffExtractor):
    """Specialized diff extractor for test files.

    Supports multiple languages through the language_specs module.
    """

    def __init__(
        self,
        repo_path: Path,
        languages: list[LanguageId | str] | None = None,
    ):
        """Initialize test diff extractor.

        Args:
            repo_path: Path to git repository
            languages: Languages to process
        """
        super().__init__(repo_path, languages=languages)

    def _classify_transform(self, diff_text: str, language_id: LanguageId | None = None) -> str | None:
        """Classify using test-specific patterns.

        First tries language-specific patterns from language_specs,
        then falls back to legacy patterns.
        """
        # Try language-specific patterns from language_specs
        if language_id:
            test_patterns = get_test_patterns(language_id)
            for transform_type, patterns in test_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, diff_text, re.MULTILINE):
                        return transform_type

        # Fall back to legacy patterns
        for transform_type, patterns in TEST_TRANSFORM_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, diff_text, re.MULTILINE):
                    return transform_type
        return None

    def _is_meaningful_transform(
        self,
        before_code: str,
        after_code: str,
        transform_type: str,
        language_id: LanguageId | None = None,
    ) -> bool:
        """Validate test transformation is meaningful.

        Args:
            before_code: Code before transformation
            after_code: Code after transformation
            transform_type: Type of transformation
            language_id: Language identifier

        Returns:
            True if transformation is meaningful
        """
        # Basic validation
        if len(before_code) < 30 or len(after_code) < 30:
            return False

        if before_code.strip() == after_code.strip():
            return False

        # Language-specific validation
        is_python = language_id == LanguageId.PYTHON if language_id else False

        if transform_type == "test-structure":
            if is_python:
                # Python: should have test function or class
                if "def test_" not in after_code and "class Test" not in after_code:
                    return False
            else:
                # TypeScript/JS: should have test blocks
                if "it(" not in after_code and "test(" not in after_code:
                    if "describe(" not in after_code:
                        return False

        if transform_type == "test-assertion":
            if is_python:
                # Python: should have assert or pytest.raises
                if "assert " not in after_code and "pytest.raises" not in after_code:
                    return False
            else:
                # TypeScript/JS: should have expect or assert
                if "expect(" not in after_code and "assert" not in after_code:
                    return False

        if transform_type == "test-mock":
            # Both languages: should have mock setup
            if is_python:
                mock_indicators = ["@patch", "MagicMock", "mocker.", "mock_"]
            else:
                mock_indicators = ["mock", "spy"]

            if not any(ind in after_code.lower() for ind in mock_indicators):
                return False

        return True


def generate_test_training_data(
    repo_path: Path,
    output_dir: Path,
    max_commits: int = 500,
    train_ratio: float = 0.9,
    languages: list[LanguageId | str] | None = None,
) -> dict[str, int]:
    """Generate test pattern training data from a repository.

    Args:
        repo_path: Path to git repository
        output_dir: Output directory for training files
        max_commits: Maximum commits to analyze
        train_ratio: Ratio of train to validation split
        languages: Languages to process (default: typescript)

    Returns:
        Statistics dictionary
    """
    generator = TestPatternGenerator(repo_path=repo_path, languages=languages)

    # Extract from git history
    pairs = generator.extract_from_git(max_commits=max_commits)
    examples = generator.generate_examples(pairs)

    if not examples:
        logger.warning("No test examples found")
        return {"total": 0, "train": 0, "valid": 0}

    # Shuffle and split
    random.shuffle(examples)
    split_idx = int(len(examples) * train_ratio)
    train_examples = examples[:split_idx]
    valid_examples = examples[split_idx:]

    # Write output files
    output_dir.mkdir(parents=True, exist_ok=True)
    generator.to_jsonl(train_examples, output_dir / "train.jsonl")
    generator.to_jsonl(valid_examples, output_dir / "valid.jsonl")

    # Also write mlx format
    generator.to_mlx_format(train_examples, output_dir / "train_mlx.jsonl")
    generator.to_mlx_format(valid_examples, output_dir / "valid_mlx.jsonl")

    stats = {
        "total": len(examples),
        "train": len(train_examples),
        "valid": len(valid_examples),
        "by_type": {},
    }

    # Count by type
    for example in examples:
        t = example.transform_type
        stats["by_type"][t] = stats["by_type"].get(t, 0) + 1

    logger.info(f"Generated test training data: {stats}")
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate test pattern training data")
    parser.add_argument("--repo", "-r", type=Path, required=True, help="Git repository path")
    parser.add_argument("--output", "-o", type=Path, default=Path("data/test-patterns"), help="Output directory")
    parser.add_argument("--max-commits", type=int, default=500, help="Maximum commits to analyze")
    parser.add_argument(
        "--languages",
        "-l",
        nargs="+",
        default=None,
        help="Languages to process (e.g., typescript python). Default: typescript",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Convert language strings to LanguageId if possible
    languages = None
    if args.languages:
        languages = []
        for lang in args.languages:
            try:
                languages.append(LanguageId(lang))
            except ValueError:
                languages.append(lang)

    stats = generate_test_training_data(
        repo_path=args.repo,
        output_dir=args.output,
        max_commits=args.max_commits,
        languages=languages,
    )

    print(f"\nGenerated {stats['total']} test examples:")
    print(f"  Train: {stats['train']}")
    print(f"  Valid: {stats['valid']}")
    print(f"  By type: {stats.get('by_type', {})}")
