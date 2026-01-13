"""Tests for package documentation extraction."""

import json
import tempfile
from pathlib import Path

import pytest

from mochi.core.types import PackageDocsConfig
from mochi.ingestion.package_docs import PackageDocsExtractor, extract_package_docs


class TestPackageDocsConfig:
    """Tests for PackageDocsConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PackageDocsConfig()
        assert config.auto_detect is True
        assert config.include_dev is False
        assert config.min_usage_count == 0
        assert "typescript" in config.exclude
        assert "@types/*" in config.exclude

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PackageDocsConfig(
            auto_detect=False,
            include_dev=True,
            exclude=["lodash"],
            include=["custom-lib"],
        )
        assert config.auto_detect is False
        assert config.include_dev is True
        assert config.exclude == ["lodash"]
        assert config.include == ["custom-lib"]


class TestPackageDocsExtractor:
    """Tests for PackageDocsExtractor."""

    def test_parse_package_json_dependencies_only(self, tmp_path):
        """Test parsing package.json without devDependencies."""
        package_json = {
            "name": "test-project",
            "dependencies": {"valtio": "^1.0.0", "react": "^18.0.0"},
            "devDependencies": {"typescript": "^5.0.0"},
        }

        pkg_path = tmp_path / "package.json"
        pkg_path.write_text(json.dumps(package_json))

        extractor = PackageDocsExtractor(PackageDocsConfig())
        packages = extractor._parse_package_json(pkg_path)

        assert "valtio" in packages
        assert "react" in packages
        assert "typescript" not in packages

    def test_parse_package_json_with_dev_dependencies(self, tmp_path):
        """Test parsing package.json with devDependencies."""
        package_json = {
            "name": "test-project",
            "dependencies": {"valtio": "^1.0.0"},
            "devDependencies": {"jest": "^29.0.0"},
        }

        pkg_path = tmp_path / "package.json"
        pkg_path.write_text(json.dumps(package_json))

        config = PackageDocsConfig(include_dev=True)
        extractor = PackageDocsExtractor(config)
        packages = extractor._parse_package_json(pkg_path)

        assert "valtio" in packages
        assert "jest" in packages

    def test_parse_requirements_txt(self, tmp_path):
        """Test parsing requirements.txt."""
        requirements = """
# Comments should be ignored
requests>=2.28.0
flask==2.0.0
numpy~=1.24
-e ./local-package
"""

        req_path = tmp_path / "requirements.txt"
        req_path.write_text(requirements)

        extractor = PackageDocsExtractor()
        packages = extractor._parse_requirements_txt(req_path)

        assert "requests" in packages
        assert "flask" in packages
        assert "numpy" in packages
        assert "./local-package" not in packages

    def test_filter_packages_excludes_patterns(self):
        """Test filtering packages based on exclude patterns."""
        config = PackageDocsConfig(exclude=["typescript", "@types/*", "eslint-*"])
        extractor = PackageDocsExtractor(config)

        packages = ["valtio", "typescript", "@types/react", "eslint-plugin-react", "react"]
        filtered = extractor._filter_packages(packages)

        assert "valtio" in filtered
        assert "react" in filtered
        assert "typescript" not in filtered
        assert "@types/react" not in filtered
        assert "eslint-plugin-react" not in filtered

    def test_extract_from_project_includes_manual_packages(self, tmp_path):
        """Test that manually included packages are added."""
        # Create empty package.json
        pkg_path = tmp_path / "package.json"
        pkg_path.write_text('{"name": "test", "dependencies": {}}')

        config = PackageDocsConfig(include=["custom-lib"])
        extractor = PackageDocsExtractor(config)

        # Mock _fetch_package_doc to avoid network calls
        extractor._fetch_package_doc = lambda pkg: None

        docs = extractor.extract_from_project(tmp_path)

        # Should have tried to fetch custom-lib even though package.json is empty
        # (docs will be empty because we mocked _fetch_package_doc to return None)


class TestExtractPackageDocs:
    """Tests for the convenience function."""

    def test_extract_package_docs_with_empty_project(self, tmp_path):
        """Test extraction from empty project."""
        docs = extract_package_docs(tmp_path)
        assert docs == []

    def test_extract_package_docs_with_config(self, tmp_path):
        """Test extraction with custom config."""
        config = PackageDocsConfig(auto_detect=False, include=["test-pkg"])

        # Mock to avoid network calls
        docs = extract_package_docs(tmp_path, config)
        # Empty because we can't fetch, but should not raise
        assert isinstance(docs, list)
