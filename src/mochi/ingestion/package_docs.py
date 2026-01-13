"""Package documentation fetcher for training data enrichment.

Extracts dependencies from package management files (package.json, pyproject.toml)
and fetches their documentation (README) for inclusion in training data.
"""

from __future__ import annotations

import fnmatch
import json
import re
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.types import PackageDocsConfig


@dataclass
class PackageDoc:
    """Documentation for a package."""

    name: str
    version: str | None
    readme: str
    source: str  # "npm", "pypi", "github", "local"


class PackageDocsExtractor:
    """Extracts and fetches package documentation."""

    def __init__(self, config: PackageDocsConfig | None = None) -> None:
        self.config = config or PackageDocsConfig()

    def extract_from_project(self, project_root: Path) -> list[PackageDoc]:
        """Extract package docs from a project.

        Args:
            project_root: Root directory of the project

        Returns:
            List of PackageDoc with fetched documentation
        """
        packages: list[str] = []

        # Try package.json (JavaScript/TypeScript)
        package_json = project_root / "package.json"
        if package_json.exists():
            packages.extend(self._parse_package_json(package_json))

        # Try pyproject.toml (Python)
        pyproject = project_root / "pyproject.toml"
        if pyproject.exists():
            packages.extend(self._parse_pyproject_toml(pyproject))

        # Try requirements.txt (Python)
        requirements = project_root / "requirements.txt"
        if requirements.exists():
            packages.extend(self._parse_requirements_txt(requirements))

        # Add manually included packages
        packages.extend(self.config.include)

        # Remove duplicates and filter
        packages = self._filter_packages(list(set(packages)))

        # Fetch documentation
        docs: list[PackageDoc] = []
        for pkg in packages:
            doc = self._fetch_package_doc(pkg)
            if doc:
                docs.append(doc)

        return docs

    def _parse_package_json(self, path: Path) -> list[str]:
        """Parse package.json and extract dependencies."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return []

        packages: list[str] = []

        # Always include dependencies
        if "dependencies" in data:
            packages.extend(data["dependencies"].keys())

        # Optionally include devDependencies
        if self.config.include_dev and "devDependencies" in data:
            packages.extend(data["devDependencies"].keys())

        return packages

    def _parse_pyproject_toml(self, path: Path) -> list[str]:
        """Parse pyproject.toml and extract dependencies."""
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return []

        packages: list[str] = []

        # Simple TOML parsing for dependencies
        # Look for [project.dependencies] or [tool.poetry.dependencies]
        in_deps_section = False
        for line in content.split("\n"):
            line = line.strip()

            if line.startswith("["):
                in_deps_section = (
                    "dependencies" in line.lower()
                    and "dev" not in line.lower()
                )
                continue

            if in_deps_section and "=" in line:
                # Extract package name (before = or >)
                pkg_name = re.split(r"[=<>~\[]", line)[0].strip().strip('"\'')
                if pkg_name and not pkg_name.startswith("#"):
                    packages.append(pkg_name)

        return packages

    def _parse_requirements_txt(self, path: Path) -> list[str]:
        """Parse requirements.txt and extract package names."""
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return []

        packages: list[str] = []
        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue

            # Extract package name (before version specifier)
            pkg_name = re.split(r"[=<>~\[;]", line)[0].strip()
            if pkg_name:
                packages.append(pkg_name)

        return packages

    def _filter_packages(self, packages: list[str]) -> list[str]:
        """Filter packages based on config."""
        filtered: list[str] = []

        for pkg in packages:
            # Check exclude patterns
            excluded = False
            for pattern in self.config.exclude:
                if fnmatch.fnmatch(pkg, pattern):
                    excluded = True
                    break

            if not excluded:
                filtered.append(pkg)

        return filtered

    def _fetch_package_doc(self, package_name: str) -> PackageDoc | None:
        """Fetch documentation for a package.

        Tries npm registry first, then falls back to PyPI.
        """
        # Try npm first (for JS/TS packages)
        doc = self._fetch_npm_readme(package_name)
        if doc:
            return doc

        # Try PyPI (for Python packages)
        doc = self._fetch_pypi_readme(package_name)
        if doc:
            return doc

        return None

    def _fetch_npm_readme(self, package_name: str) -> PackageDoc | None:
        """Fetch README from npm registry."""
        # Handle scoped packages (@org/name)
        if package_name.startswith("@"):
            url_name = package_name.replace("/", "%2F")
        else:
            url_name = package_name

        url = f"https://registry.npmjs.org/{url_name}"

        try:
            req = urllib.request.Request(
                url,
                headers={"Accept": "application/json", "User-Agent": "mochi/1.0"},
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
            return None

        readme = data.get("readme", "")
        if not readme:
            return None

        # Truncate if too long
        if len(readme) > self.config.max_doc_size:
            readme = readme[: self.config.max_doc_size] + "\n\n[Truncated]"

        version = data.get("dist-tags", {}).get("latest")

        return PackageDoc(
            name=package_name,
            version=version,
            readme=readme,
            source="npm",
        )

    def _fetch_pypi_readme(self, package_name: str) -> PackageDoc | None:
        """Fetch README from PyPI."""
        url = f"https://pypi.org/pypi/{package_name}/json"

        try:
            req = urllib.request.Request(
                url,
                headers={"Accept": "application/json", "User-Agent": "mochi/1.0"},
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
            return None

        info = data.get("info", {})
        description = info.get("description", "")

        if not description:
            return None

        # Truncate if too long
        if len(description) > self.config.max_doc_size:
            description = description[: self.config.max_doc_size] + "\n\n[Truncated]"

        version = info.get("version")

        return PackageDoc(
            name=package_name,
            version=version,
            readme=description,
            source="pypi",
        )


def extract_package_docs(
    project_root: Path,
    config: PackageDocsConfig | None = None,
) -> list[PackageDoc]:
    """Convenience function to extract package docs from a project.

    Args:
        project_root: Root directory of the project
        config: Configuration for extraction

    Returns:
        List of PackageDoc with fetched documentation
    """
    extractor = PackageDocsExtractor(config)
    return extractor.extract_from_project(project_root)
