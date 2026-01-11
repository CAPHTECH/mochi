"""Extract code transformation pairs from git diff history.

This module extracts before/after code pairs from git commits to create
training data for code transformation tasks (error-handling, null-safety, etc.).

Works with any TypeScript/JavaScript repository.
"""

from __future__ import annotations

import fnmatch
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

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


# Patterns to classify transformation types
TRANSFORM_PATTERNS: dict[str, list[str]] = {
    "error-handling": [
        r"\+.*try\s*\{",
        r"\+.*catch\s*\(",
        r"\+.*throw\s+new",
        r"\+.*Result<",
        r"\+.*\.catch\s*\(",
        r"\+.*finally\s*\{",
    ],
    "null-safety": [
        r"\+.*\?\.",
        r"\+.*\?\?",
        r"\+.*!= null",
        r"\+.*!== null",
        r"\+.*!= undefined",
        r"\+.*!== undefined",
    ],
    "type-safety": [
        r"\+.*:\s*[A-Z][a-zA-Z0-9_]+\s*[=;]",
        r"\+.*as\s+[A-Z][a-zA-Z0-9_]+",
        r"\+.*is\s+[A-Z][a-zA-Z0-9_]+",
        r"\+.*<[A-Z][a-zA-Z0-9_]+>",
    ],
    "async-await": [
        r"\+.*async\s+function",
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
}


class GitDiffExtractor:
    """Extract code transformation pairs from git history."""

    def __init__(self, repo_path: Path):
        """Initialize extractor for a git repository.

        Args:
            repo_path: Path to the git repository root
        """
        self.repo_path = Path(repo_path).resolve()
        if not (self.repo_path / ".git").exists():
            raise ValueError(f"Not a git repository: {repo_path}")

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
            file_patterns: Glob patterns for files to include (default: ["*.ts", "*.tsx"])
            max_commits: Maximum number of commits to analyze
            transform_types: Types of transformations to extract (default: all)
            min_lines_changed: Minimum lines changed to consider (filters noise)
            max_lines_changed: Maximum lines changed to consider (filters large refactors)

        Returns:
            List of CodeTransformPair instances
        """
        if file_patterns is None:
            file_patterns = ["*.ts", "*.tsx"]

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
        # Count changed lines
        added_lines = len([l for l in diff_text.split("\n") if l.startswith("+")])
        removed_lines = len([l for l in diff_text.split("\n") if l.startswith("-")])
        total_changed = added_lines + removed_lines

        if total_changed < min_lines_changed or total_changed > max_lines_changed:
            return None

        # Classify the transformation
        transform_type = self._classify_transform(diff_text)
        if not transform_type:
            return None

        if transform_types and transform_type not in transform_types:
            return None

        # Extract before/after code
        before_code, after_code = self._extract_before_after(
            commit_hash, file_path, diff_text
        )

        if not before_code or not after_code:
            return None

        # Validate the transformation is meaningful
        if not self._is_meaningful_transform(before_code, after_code, transform_type):
            return None

        return CodeTransformPair(
            file_path=file_path,
            before_code=before_code,
            after_code=after_code,
            commit_hash=commit_hash,
            commit_message=commit_message,
            transform_type=transform_type,
            diff_text=diff_text,
        )

    def _classify_transform(self, diff_text: str) -> str | None:
        """Classify the type of code transformation from diff text."""
        for transform_type, patterns in TRANSFORM_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, diff_text, re.MULTILINE):
                    return transform_type
        return None

    def _extract_before_after(
        self, commit_hash: str, file_path: str, diff_text: str
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
            result_before.stdout, diff_text, is_before=True
        )
        after_code = self._extract_context_block(
            result_after.stdout, diff_text, is_before=False
        )

        return before_code, after_code

    def _extract_context_block(
        self, full_content: str, diff_text: str, is_before: bool
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

        # Find the enclosing function/block
        block_start = self._find_block_start(lines, start_line)
        block_end = self._find_block_end(lines, start_line)

        # Extract the block with some context
        extracted_lines = lines[max(0, block_start - 1) : min(len(lines), block_end + 1)]

        return "\n".join(extracted_lines)

    def _find_block_start(self, lines: list[str], target_line: int) -> int:
        """Find the start of the enclosing function/class/block."""
        # Look for function/class/const declarations
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

    def _find_block_end(self, lines: list[str], target_line: int) -> int:
        """Find the end of the enclosing function/class/block."""
        # Simple brace counting from target line
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

    def _is_meaningful_transform(
        self, before_code: str, after_code: str, transform_type: str
    ) -> bool:
        """Validate that the transformation is meaningful for learning."""
        # Check minimum content
        if len(before_code) < 50 or len(after_code) < 50:
            return False

        # Check that there's actual difference
        if before_code.strip() == after_code.strip():
            return False

        # Type-specific validation
        if transform_type == "error-handling":
            # After code should have try/catch
            if "try" not in after_code and "catch" not in after_code:
                return False
            # Before code should NOT have try/catch (transformation added it)
            if "try {" in before_code:
                return False

        if transform_type == "null-safety":
            # After code should have optional chaining or null checks
            if "?." not in after_code and "??" not in after_code:
                if "!= null" not in after_code and "!== null" not in after_code:
                    return False

        if transform_type == "async-await":
            # After code should have async/await
            if "async" not in after_code and "await" not in after_code:
                return False

        return True

    def _matches_patterns(self, file_path: str, patterns: list[str]) -> bool:
        """Check if file path matches any of the glob patterns."""
        return any(fnmatch.fnmatch(file_path, p) for p in patterns)


def extract_transforms_from_repos(
    repo_paths: list[Path],
    file_patterns: list[str] | None = None,
    transform_types: list[str] | None = None,
    max_commits_per_repo: int = 500,
) -> list[CodeTransformPair]:
    """Extract transformation pairs from multiple repositories.

    Args:
        repo_paths: List of paths to git repositories
        file_patterns: Glob patterns for files (default: ["*.ts", "*.tsx"])
        transform_types: Types to extract (default: all)
        max_commits_per_repo: Max commits to analyze per repo

    Returns:
        Combined list of transformation pairs from all repos
    """
    all_pairs: list[CodeTransformPair] = []

    for repo_path in repo_paths:
        try:
            extractor = GitDiffExtractor(repo_path)
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
