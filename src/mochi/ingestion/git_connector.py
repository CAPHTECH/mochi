"""Git repository connector for code ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from git import Repo


@dataclass
class SourceFile:
    """Represents a source file from the repository."""

    path: str
    content: str
    language: str


class GitConnector:
    """Connects to Git repositories and extracts source files."""

    LANGUAGE_EXTENSIONS: dict[str, str] = {
        # Programming languages
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".py": "python",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".swift": "swift",
        ".kt": "kotlin",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "csharp",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        # Data/Config formats
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".xml": "xml",
        ".sql": "sql",
        ".graphql": "graphql",
        ".gql": "graphql",
        # Documentation/Text
        ".md": "markdown",
        ".mdx": "markdown",
        ".txt": "text",
        ".rst": "restructuredtext",
        # Shell/Scripts
        ".sh": "shell",
        ".bash": "shell",
        ".zsh": "shell",
        ".fish": "shell",
        ".ps1": "powershell",
        # Web
        ".html": "html",
        ".htm": "html",
        ".css": "css",
        ".scss": "scss",
        ".sass": "sass",
        ".less": "less",
        ".vue": "vue",
        ".svelte": "svelte",
        # Test/Spec
        ".feature": "gherkin",
        ".spec": "text",
        # Other
        ".dockerfile": "dockerfile",
        ".env": "dotenv",
        ".gitignore": "gitignore",
    }

    def __init__(self, repo_path: str | Path) -> None:
        """Initialize with a local repository path."""
        self.repo_path = Path(repo_path)
        self.repo = Repo(self.repo_path)

    @classmethod
    def clone(cls, url: str, target_path: str | Path) -> "GitConnector":
        """Clone a repository and return a connector."""
        target = Path(target_path)
        if target.exists():
            return cls(target)
        Repo.clone_from(url, target)
        return cls(target)

    def get_source_files(
        self,
        extensions: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> list[SourceFile]:
        """
        Get all source files from the repository.

        Args:
            extensions: List of file extensions to include (e.g., [".ts", ".py"])
            exclude_patterns: Glob patterns to exclude (e.g., ["**/test/**", "**/node_modules/**"])
        """
        if extensions is None:
            extensions = list(self.LANGUAGE_EXTENSIONS.keys())

        if exclude_patterns is None:
            exclude_patterns = [
                "**/node_modules/**",
                "**/dist/**",
                "**/build/**",
                "**/.git/**",
                "**/vendor/**",
                "**/__pycache__/**",
            ]

        files: list[SourceFile] = []

        for ext in extensions:
            for file_path in self.repo_path.rglob(f"*{ext}"):
                # Check exclude patterns
                rel_path = file_path.relative_to(self.repo_path)
                if self._should_exclude(rel_path, exclude_patterns):
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8")
                    language = self.LANGUAGE_EXTENSIONS.get(ext, "unknown")
                    files.append(
                        SourceFile(
                            path=str(rel_path),
                            content=content,
                            language=language,
                        )
                    )
                except (UnicodeDecodeError, PermissionError):
                    continue

        return files

    def _should_exclude(self, path: Path, patterns: list[str]) -> bool:
        """Check if a path should be excluded based on patterns."""
        path_str = str(path)
        for pattern in patterns:
            # Simple glob-like matching
            if "**" in pattern:
                # Match any path containing the pattern part
                pattern_part = pattern.replace("**/", "").replace("/**", "")
                if pattern_part in path_str:
                    return True
            elif path_str.startswith(pattern) or path_str.endswith(pattern):
                return True
        return False

    def get_file_count(self, extensions: list[str] | None = None) -> int:
        """Get the count of source files."""
        return len(self.get_source_files(extensions))
