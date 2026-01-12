"""Extract code transformation pairs from git diff history.

This module extracts before/after code pairs from git commits to create
training data for code transformation tasks (error-handling, null-safety, etc.).

Supports multiple languages through language_specs configuration.
"""

from __future__ import annotations

import fnmatch
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from mochi.core.language_specs import (
    LanguageId,
    LanguageSpec,
    LANGUAGE_SPECS,
    detect_language,
    get_file_patterns_for_languages,
)

logger = logging.getLogger(__name__)


@dataclass
class CodeTransformPair:
    """Before/after code pair from git diff."""

    file_path: str
    before_code: str
    after_code: str
    commit_hash: str
    commit_message: str
    transform_type: str  # error-handling, null-safety, etc.
    diff_text: str  # Raw diff for debugging
    language: LanguageId | None = None  # Detected language


# Legacy patterns for backward compatibility - now prefer language_specs
# These are used as fallback when language detection fails
_LEGACY_TRANSFORM_PATTERNS: dict[str, list[str]] = {
    "error-handling": [
        r"\+.*try\s*[\{:]",
        r"\+.*catch\s*\(",
        r"\+.*except\s+",
        r"\+.*throw\s+new",
        r"\+.*raise\s+",
        r"\+.*Result<",
        r"\+.*\.catch\s*\(",
        r"\+.*finally\s*[\{:]",
    ],
    "null-safety": [
        r"\+.*\?\.",
        r"\+.*\?\?",
        r"\+.*!= null",
        r"\+.*!== null",
        r"\+.*!= undefined",
        r"\+.*!== undefined",
        r"\+.*is\s+None",
        r"\+.*is\s+not\s+None",
    ],
    "type-safety": [
        r"\+.*:\s*[A-Z][a-zA-Z0-9_]+\s*[=;]",
        r"\+.*as\s+[A-Z][a-zA-Z0-9_]+",
        r"\+.*is\s+[A-Z][a-zA-Z0-9_]+",
        r"\+.*<[A-Z][a-zA-Z0-9_]+>",
        r"\+.*->\s*[A-Z][a-zA-Z0-9_]+",
    ],
    "async-await": [
        r"\+.*async\s+function",
        r"\+.*async\s+def",
        r"\+.*async\s*\(",
        r"\+.*await\s+",
    ],
    "validation": [
        r"\+.*\.parse\s*\(",
        r"\+.*\.safeParse\s*\(",
        r"\+.*validate\w*\s*\(",
        r"\+.*z\.\w+\s*\(",
        r"\+.*assert\w*\s*\(",
    ],
    # Test patterns (combined from multiple frameworks)
    "test-structure": [
        r"\+\s*describe\s*\(",
        r"\+\s*it\s*\(",
        r"\+\s*test\s*\(",
        r"\+\s*expect\s*\(",
        r"\+\s*def\s+test_\w+",
        r"\+\s*class\s+Test\w+",
    ],
    "test-assertion": [
        r"\+.*expect\s*\([^)]+\)\s*\.\w+",
        r"\+.*\.toBe\s*\(",
        r"\+.*\.toEqual\s*\(",
        r"\+.*\.toHaveBeenCalled",
        r"\+.*\.toThrow\s*\(",
        r"\+.*\.toMatchSnapshot\s*\(",
        r"\+\s*assert\s+",
        r"\+.*self\.assert\w+\s*\(",
    ],
    "test-setup": [
        r"\+\s*beforeEach\s*\(",
        r"\+\s*afterEach\s*\(",
        r"\+\s*beforeAll\s*\(",
        r"\+\s*afterAll\s*\(",
        r"\+\s*@pytest\.fixture",
        r"\+\s*def\s+setUp\s*\(",
    ],
    "test-mock": [
        r"\+\s*vi\.mock\s*\(",
        r"\+\s*jest\.mock\s*\(",
        r"\+\s*vi\.fn\s*\(",
        r"\+\s*jest\.fn\s*\(",
        r"\+.*\.mockResolvedValue\s*\(",
        r"\+.*\.mockReturnValue\s*\(",
        r"\+.*\.mockImplementation\s*\(",
        r"\+\s*@patch\s*\(",
        r"\+\s*MagicMock\s*\(",
    ],
}

# Backward compatibility alias
TRANSFORM_PATTERNS = _LEGACY_TRANSFORM_PATTERNS


class GitDiffExtractor:
    """Extract code transformation pairs from git history.

    Supports multiple languages through language_specs configuration.
    By default, extracts from TypeScript only for backward compatibility.
    """

    def __init__(
        self,
        repo_path: Path,
        languages: list[LanguageId | str] | None = None,
    ):
        """Initialize extractor for a git repository.

        Args:
            repo_path: Path to the git repository root
            languages: List of languages to extract from.
                       If None, defaults to TypeScript only for backward compatibility.
        """
        self.repo_path = Path(repo_path).resolve()
        if not (self.repo_path / ".git").exists():
            raise ValueError(f"Not a git repository: {repo_path}")

        # Normalize language IDs
        # Default to TypeScript only for backward compatibility
        if languages is None:
            self.languages: list[LanguageId] = [LanguageId.TYPESCRIPT]
        else:
            self.languages = []
            for lang in languages:
                if isinstance(lang, str):
                    try:
                        self.languages.append(LanguageId(lang))
                    except ValueError:
                        logger.warning(f"Unknown language: {lang}, skipping")
                else:
                    self.languages.append(lang)

    def extract_transforms(
        self,
        file_patterns: list[str] | None = None,
        max_commits: int = 1000,
        transform_types: list[str] | None = None,
        min_lines_changed: int = 3,
        max_lines_changed: int = 100,
    ) -> list[CodeTransformPair]:
        """Extract transformation pairs from commit history.

        Args:
            file_patterns: Glob patterns for files to include. If None, uses patterns
                          from configured languages.
            max_commits: Maximum number of commits to analyze
            transform_types: Types of transformations to extract (default: all)
            min_lines_changed: Minimum lines changed to consider (filters noise)
            max_lines_changed: Maximum lines changed to consider (filters large refactors)

        Returns:
            List of CodeTransformPair instances
        """
        if file_patterns is None:
            file_patterns = get_file_patterns_for_languages(self.languages)

        pairs: list[CodeTransformPair] = []

        for commit_hash, commit_message in self._iter_commits(max_commits):
            try:
                commit_pairs = self._extract_from_commit(
                    commit_hash,
                    commit_message,
                    file_patterns,
                    transform_types,
                    min_lines_changed,
                    max_lines_changed,
                )
                pairs.extend(commit_pairs)
            except Exception as e:
                logger.debug(f"Error processing commit {commit_hash[:8]}: {e}")
                continue

        logger.info(f"Extracted {len(pairs)} transformation pairs from {self.repo_path}")
        return pairs

    def _iter_commits(self, max_commits: int) -> Iterator[tuple[str, str]]:
        """Iterate through commits (excluding merges).

        Yields:
            Tuples of (commit_hash, commit_message)
        """
        result = subprocess.run(
            [
                "git",
                "log",
                "--no-merges",
                "--format=%H|%s",
                f"-{max_commits}",
            ],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"Git log failed: {result.stderr}")
            return

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|", 1)
            if len(parts) == 2:
                yield parts[0], parts[1]

    def _extract_from_commit(
        self,
        commit_hash: str,
        commit_message: str,
        file_patterns: list[str],
        transform_types: list[str] | None,
        min_lines_changed: int,
        max_lines_changed: int,
    ) -> list[CodeTransformPair]:
        """Extract transformation pairs from a single commit."""
        pairs: list[CodeTransformPair] = []

        # Get diff for this commit
        result = subprocess.run(
            [
                "git",
                "diff",
                f"{commit_hash}^",
                commit_hash,
                "--",
            ],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return pairs

        # Parse the diff
        current_file = None
        current_diff_lines: list[str] = []

        for line in result.stdout.split("\n"):
            if line.startswith("diff --git"):
                # Process previous file's diff
                if current_file and current_diff_lines:
                    pair = self._process_file_diff(
                        current_file,
                        "\n".join(current_diff_lines),
                        commit_hash,
                        commit_message,
                        transform_types,
                        min_lines_changed,
                        max_lines_changed,
                    )
                    if pair:
                        pairs.append(pair)

                # Extract file path from diff line
                # Format: "diff --git a/path/to/file b/path/to/file"
                match = re.search(r"diff --git a/(.+?) b/(.+)", line)
                if match:
                    file_path = match.group(2)
                    if self._matches_patterns(file_path, file_patterns):
                        current_file = file_path
                        current_diff_lines = []
                    else:
                        current_file = None
                else:
                    current_file = None
            elif current_file:
                current_diff_lines.append(line)

        # Process last file
        if current_file and current_diff_lines:
            pair = self._process_file_diff(
                current_file,
                "\n".join(current_diff_lines),
                commit_hash,
                commit_message,
                transform_types,
                min_lines_changed,
                max_lines_changed,
            )
            if pair:
                pairs.append(pair)

        return pairs

    def _process_file_diff(
        self,
        file_path: str,
        diff_text: str,
        commit_hash: str,
        commit_message: str,
        transform_types: list[str] | None,
        min_lines_changed: int,
        max_lines_changed: int,
    ) -> CodeTransformPair | None:
        """Process a single file's diff and create a transformation pair."""
        # Detect language from file path
        language_id = detect_language(file_path)

        # Count changed lines
        added_lines = len([l for l in diff_text.split("\n") if l.startswith("+")])
        removed_lines = len([l for l in diff_text.split("\n") if l.startswith("-")])
        total_changed = added_lines + removed_lines

        if total_changed < min_lines_changed or total_changed > max_lines_changed:
            return None

        # Classify the transformation using language-specific patterns
        transform_type = self._classify_transform(diff_text, language_id)
        if not transform_type:
            return None

        if transform_types and transform_type not in transform_types:
            return None

        # Extract before/after code
        before_code, after_code = self._extract_before_after(
            commit_hash, file_path, diff_text, language_id
        )

        if not before_code or not after_code:
            return None

        # Validate the transformation is meaningful
        if not self._is_meaningful_transform(
            before_code, after_code, transform_type, language_id
        ):
            return None

        return CodeTransformPair(
            file_path=file_path,
            before_code=before_code,
            after_code=after_code,
            commit_hash=commit_hash,
            commit_message=commit_message,
            transform_type=transform_type,
            diff_text=diff_text,
            language=language_id,
        )

    def _classify_transform(
        self, diff_text: str, language_id: LanguageId | None = None
    ) -> str | None:
        """Classify the type of code transformation from diff text.

        Uses language-specific patterns when language is known,
        falls back to legacy patterns otherwise.
        """
        patterns_to_check: dict[str, tuple[str, ...] | list[str]]

        if language_id is not None and language_id in LANGUAGE_SPECS:
            # Use language-specific patterns
            spec = LANGUAGE_SPECS[language_id]
            patterns_to_check = spec.get_all_transform_patterns()
        else:
            # Fallback to legacy patterns
            patterns_to_check = _LEGACY_TRANSFORM_PATTERNS

        for transform_type, patterns in patterns_to_check.items():
            for pattern in patterns:
                if re.search(pattern, diff_text, re.MULTILINE):
                    return transform_type
        return None

    def _extract_before_after(
        self,
        commit_hash: str,
        file_path: str,
        diff_text: str,
        language_id: LanguageId | None = None,
    ) -> tuple[str, str]:
        """Extract before and after code from git history.

        Returns the relevant code chunks around the changed lines.
        """
        # Get file content before the commit
        result_before = subprocess.run(
            ["git", "show", f"{commit_hash}^:{file_path}"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )

        # Get file content after the commit
        result_after = subprocess.run(
            ["git", "show", f"{commit_hash}:{file_path}"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )

        if result_before.returncode != 0 or result_after.returncode != 0:
            return "", ""

        # Extract the relevant function/block around the changes
        before_code = self._extract_context_block(
            result_before.stdout, diff_text, is_before=True, language_id=language_id
        )
        after_code = self._extract_context_block(
            result_after.stdout, diff_text, is_before=False, language_id=language_id
        )

        return before_code, after_code

    def _extract_context_block(
        self,
        full_content: str,
        diff_text: str,
        is_before: bool,
        language_id: LanguageId | None = None,
    ) -> str:
        """Extract the relevant code block around the changed lines.

        Attempts to extract a complete function or block rather than just
        the changed lines for better training context.
        """
        # Parse diff hunk headers to find line numbers
        # Format: @@ -start,count +start,count @@
        hunk_pattern = re.compile(r"@@ -(\d+),?\d* \+(\d+),?\d* @@")
        hunks = hunk_pattern.findall(diff_text)

        if not hunks:
            return ""

        # Get the first changed line number
        if is_before:
            start_line = int(hunks[0][0])
        else:
            start_line = int(hunks[0][1])

        lines = full_content.split("\n")

        # Find the enclosing function/block using language-specific patterns
        block_start = self._find_block_start(lines, start_line, language_id)
        block_end = self._find_block_end(lines, start_line, language_id)

        # Extract the block with some context
        extracted_lines = lines[max(0, block_start - 1) : min(len(lines), block_end + 1)]

        return "\n".join(extracted_lines)

    def _find_block_start(
        self, lines: list[str], target_line: int, language_id: LanguageId | None = None
    ) -> int:
        """Find the start of the enclosing function/class/block.

        Uses language-specific block patterns when available.
        """
        # Get language-specific block patterns
        block_patterns: list[str] = []
        if language_id is not None and language_id in LANGUAGE_SPECS:
            spec = LANGUAGE_SPECS[language_id]
            block_patterns = [bp.start_pattern for bp in spec.block_patterns]
        else:
            # Fallback to legacy patterns (TypeScript-style)
            block_patterns = [
                r"^\s*(?:export\s+)?(?:async\s+)?function\s+\w+",
                r"^\s*(?:export\s+)?class\s+\w+",
                r"^\s*(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\(",
                r"^\s*(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*\{",
            ]

        for i in range(target_line - 1, -1, -1):
            if i >= len(lines):
                continue
            line = lines[i]
            for pattern in block_patterns:
                if re.match(pattern, line):
                    return i

        # If no block found, return some context before
        return max(0, target_line - 10)

    def _find_block_end(
        self, lines: list[str], target_line: int, language_id: LanguageId | None = None
    ) -> int:
        """Find the end of the enclosing function/class/block.

        Uses different strategies based on language:
        - Brace-based (TypeScript, JavaScript): count { and }
        - Indentation-based (Python): track indentation level
        """
        # Check if language uses indentation for blocks
        uses_indentation = False
        if language_id is not None and language_id in LANGUAGE_SPECS:
            spec = LANGUAGE_SPECS[language_id]
            uses_indentation = any(bp.uses_indentation for bp in spec.block_patterns)

        if uses_indentation:
            return self._find_block_end_by_indentation(lines, target_line)
        else:
            return self._find_block_end_by_braces(lines, target_line)

    def _find_block_end_by_braces(self, lines: list[str], target_line: int) -> int:
        """Find block end using brace counting (for C-style languages)."""
        brace_count = 0
        found_open = False

        for i in range(target_line - 1, len(lines)):
            if i < 0 or i >= len(lines):
                continue
            line = lines[i]
            brace_count += line.count("{") - line.count("}")

            if "{" in line:
                found_open = True

            if found_open and brace_count <= 0:
                return i + 1

        # If no end found, return some context after
        return min(len(lines), target_line + 20)

    def _find_block_end_by_indentation(self, lines: list[str], target_line: int) -> int:
        """Find block end using indentation (for Python-style languages).

        Block ends when we reach a line with same or lower indentation
        than the block start.
        """
        if target_line <= 0 or target_line >= len(lines):
            return min(len(lines), target_line + 20)

        # Find the start line's indentation (look for def/class)
        start_indent = 0
        for i in range(target_line - 1, -1, -1):
            line = lines[i]
            stripped = line.lstrip()
            if stripped.startswith(("def ", "async def ", "class ")):
                start_indent = len(line) - len(stripped)
                break

        # Find where indentation returns to or below start level
        for i in range(target_line, len(lines)):
            line = lines[i]
            # Skip empty lines and comments
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            current_indent = len(line) - len(line.lstrip())
            # Block ends when we return to same or lower indentation
            # (but not for the first line of the block)
            if i > target_line and current_indent <= start_indent:
                return i

        return len(lines)

    def _is_meaningful_transform(
        self,
        before_code: str,
        after_code: str,
        transform_type: str,
        language_id: LanguageId | None = None,
    ) -> bool:
        """Validate that the transformation is meaningful for learning.

        Uses language-specific validation when language is known.
        """
        # Check minimum content
        if len(before_code) < 50 or len(after_code) < 50:
            return False

        # Check that there's actual difference
        if before_code.strip() == after_code.strip():
            return False

        # Language-specific keywords
        is_python = language_id == LanguageId.PYTHON

        # Type-specific validation
        if transform_type == "error-handling":
            if is_python:
                # Python: try/except
                if "try:" not in after_code and "except" not in after_code:
                    return False
                if "try:" in before_code:
                    return False
            else:
                # TypeScript/JavaScript: try/catch
                if "try" not in after_code and "catch" not in after_code:
                    return False
                if "try {" in before_code:
                    return False

        if transform_type == "null-safety":
            if is_python:
                # Python: is None, is not None
                has_null_check = any(
                    check in after_code
                    for check in ["is None", "is not None", "if not ", "if "]
                )
                if not has_null_check:
                    return False
            else:
                # TypeScript/JavaScript: ?., ??
                if "?." not in after_code and "??" not in after_code:
                    if "!= null" not in after_code and "!== null" not in after_code:
                        return False

        if transform_type == "async-await":
            # async/await works similarly in both languages
            if "async" not in after_code and "await" not in after_code:
                return False

        # Test patterns validation (language-specific)
        if transform_type == "test-structure":
            if is_python:
                # Python: def test_, class Test
                has_test_structure = any(
                    keyword in after_code
                    for keyword in ["def test_", "class Test", "@pytest"]
                )
            else:
                # TypeScript/JavaScript: describe/it/test/expect
                has_test_structure = any(
                    keyword in after_code
                    for keyword in ["describe(", "it(", "test(", "expect("]
                )
            if not has_test_structure:
                return False

        if transform_type == "test-assertion":
            if is_python:
                # Python: assert, pytest.raises
                has_assertion = "assert " in after_code or "pytest.raises" in after_code
                if not has_assertion:
                    # Check for unittest style
                    has_assertion = any(
                        f"self.assert" in after_code or f".assert" in after_code
                        for _ in [1]
                    )
                if not has_assertion:
                    return False
            else:
                # TypeScript/JavaScript: expect with matcher
                if "expect(" not in after_code:
                    return False
                has_matcher = any(
                    matcher in after_code
                    for matcher in [
                        ".toBe(",
                        ".toEqual(",
                        ".toHaveBeenCalled",
                        ".toThrow(",
                        ".toMatch",
                        ".toContain(",
                    ]
                )
                if not has_matcher:
                    return False

        if transform_type == "test-setup":
            if is_python:
                # Python: @pytest.fixture, setUp
                has_hook = any(
                    hook in after_code
                    for hook in [
                        "@pytest.fixture",
                        "def setup",
                        "def setUp",
                        "def teardown",
                        "def tearDown",
                    ]
                )
            else:
                # TypeScript/JavaScript: beforeEach, etc.
                has_hook = any(
                    hook in after_code
                    for hook in ["beforeEach(", "afterEach(", "beforeAll(", "afterAll("]
                )
            if not has_hook:
                return False

        if transform_type == "test-mock":
            if is_python:
                # Python: @patch, Mock, MagicMock
                has_mock = any(
                    mock in after_code
                    for mock in [
                        "@patch",
                        "Mock(",
                        "MagicMock(",
                        "AsyncMock(",
                        "mocker.",
                        "monkeypatch.",
                    ]
                )
            else:
                # TypeScript/JavaScript: vi.mock, jest.mock
                has_mock = any(
                    mock in after_code
                    for mock in [
                        "vi.mock(",
                        "jest.mock(",
                        "vi.fn(",
                        "jest.fn(",
                        ".mockResolvedValue(",
                        ".mockReturnValue(",
                    ]
                )
            if not has_mock:
                return False

        return True

    def _matches_patterns(self, file_path: str, patterns: list[str]) -> bool:
        """Check if file path matches any of the glob patterns."""
        return any(fnmatch.fnmatch(file_path, p) for p in patterns)


def extract_transforms_from_repos(
    repo_paths: list[Path],
    languages: list[LanguageId | str] | None = None,
    file_patterns: list[str] | None = None,
    transform_types: list[str] | None = None,
    max_commits_per_repo: int = 500,
) -> list[CodeTransformPair]:
    """Extract transformation pairs from multiple repositories.

    Args:
        repo_paths: List of paths to git repositories
        languages: Languages to extract from (default: all supported languages)
        file_patterns: Glob patterns for files (default: derived from languages)
        transform_types: Types to extract (default: all)
        max_commits_per_repo: Max commits to analyze per repo

    Returns:
        Combined list of transformation pairs from all repos
    """
    all_pairs: list[CodeTransformPair] = []

    for repo_path in repo_paths:
        try:
            extractor = GitDiffExtractor(repo_path, languages=languages)
            pairs = extractor.extract_transforms(
                file_patterns=file_patterns,
                max_commits=max_commits_per_repo,
                transform_types=transform_types,
            )
            all_pairs.extend(pairs)
            logger.info(f"Extracted {len(pairs)} pairs from {repo_path}")
        except Exception as e:
            logger.error(f"Failed to process {repo_path}: {e}")

    return all_pairs
