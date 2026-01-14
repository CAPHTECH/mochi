"""Mochi package format and utilities.

.mochi package structure:
    my-project.mochi/
    ├── manifest.json       # Package metadata
    ├── adapter_config.json # Adapter configuration
    └── adapters.safetensors # LoRA weights

manifest.json:
{
    "name": "my-project",
    "version": "1.0.0",
    "base_model": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
    "adapter_type": "project",
    "description": "Project adapter for my-project",
    "created_at": "2024-01-12T10:00:00Z",
    "mochi_version": "0.1.0"
}
"""

from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlretrieve

# Default paths
DEFAULT_MOCHI_DIR = Path.home() / ".mochi"
DEFAULT_ADAPTERS_DIR = DEFAULT_MOCHI_DIR / "adapters"
DEFAULT_CACHE_DIR = DEFAULT_MOCHI_DIR / "cache"

# Current mochi version
MOCHI_VERSION = "0.1.0"


@dataclass
class PackageManifest:
    """Manifest for a .mochi package."""

    name: str
    version: str
    base_model: str
    adapter_type: str  # "base" or "project"
    description: str = ""
    created_at: str = ""
    mochi_version: str = MOCHI_VERSION
    # Optional fields
    patterns: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    base_adapter: str | None = None  # For project adapters
    train_examples: int = 0
    final_loss: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PackageManifest:
        """Create manifest from dictionary."""
        return cls(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            base_model=data["base_model"],
            adapter_type=data.get("adapter_type", "project"),
            description=data.get("description", ""),
            created_at=data.get("created_at", ""),
            mochi_version=data.get("mochi_version", MOCHI_VERSION),
            patterns=data.get("patterns", []),
            languages=data.get("languages", []),
            base_adapter=data.get("base_adapter"),
            train_examples=data.get("train_examples", 0),
            final_loss=data.get("final_loss"),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v}

    def save(self, path: Path) -> None:
        """Save manifest to file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> PackageManifest:
        """Load manifest from file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


class MochiPackage:
    """A .mochi package containing an adapter."""

    def __init__(self, path: Path) -> None:
        """Initialize package from path.

        Args:
            path: Path to .mochi directory or .mochi.tar.gz archive
        """
        self.path = Path(path)
        self._manifest: PackageManifest | None = None
        self._is_archive = self.path.suffix in (".gz", ".tar")

    @property
    def manifest(self) -> PackageManifest:
        """Get package manifest."""
        if self._manifest is None:
            if self._is_archive:
                self._manifest = self._read_manifest_from_archive()
            else:
                manifest_path = self.path / "manifest.json"
                if manifest_path.exists():
                    self._manifest = PackageManifest.load(manifest_path)
                else:
                    # Try adapter_config.json for backwards compatibility
                    config_path = self.path / "adapter_config.json"
                    if config_path.exists():
                        self._manifest = self._manifest_from_adapter_config(config_path)
                    else:
                        raise FileNotFoundError(
                            f"No manifest.json or adapter_config.json in {self.path}"
                        )
        return self._manifest

    def _manifest_from_adapter_config(self, config_path: Path) -> PackageManifest:
        """Create manifest from legacy adapter_config.json."""
        with open(config_path) as f:
            config = json.load(f)

        return PackageManifest(
            name=config.get("name", self.path.name),
            version="1.0.0",
            base_model=config.get("base_model", "unknown"),
            adapter_type=config.get("type", "project"),
            patterns=config.get("patterns", []),
            languages=config.get("languages", []),
            train_examples=config.get("train_examples", 0),
            final_loss=config.get("final_val_loss"),
        )

    def _read_manifest_from_archive(self) -> PackageManifest:
        """Read manifest from tar.gz archive without full extraction."""
        with tarfile.open(self.path, "r:gz") as tar:
            # Find manifest.json
            for member in tar.getmembers():
                if member.name.endswith("manifest.json"):
                    f = tar.extractfile(member)
                    if f:
                        return PackageManifest.from_dict(json.load(f))
            raise FileNotFoundError(f"No manifest.json in archive {self.path}")

    @property
    def adapter_path(self) -> Path:
        """Get path to adapter weights."""
        if self._is_archive:
            raise ValueError("Cannot get adapter path from archive. Extract first.")
        # Check for adapter subdirectory
        adapter_dir = self.path / "adapter"
        if adapter_dir.exists():
            return adapter_dir
        # Check for weights directly in package
        weights_file = self.path / "adapters.safetensors"
        if weights_file.exists():
            return self.path
        raise FileNotFoundError(f"No adapter weights found in {self.path}")

    def extract_to(self, target_dir: Path) -> Path:
        """Extract archive to target directory.

        Args:
            target_dir: Directory to extract to

        Returns:
            Path to extracted package directory
        """
        if not self._is_archive:
            raise ValueError("Package is not an archive")

        target_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(self.path, "r:gz") as tar:
            # Get root directory name from archive
            root_name = None
            for member in tar.getmembers():
                parts = member.name.split("/")
                if parts[0]:
                    root_name = parts[0]
                    break

            tar.extractall(target_dir)

        if root_name:
            return target_dir / root_name
        return target_dir


def pack_adapter(
    adapter_dir: Path,
    output_path: Path | None = None,
    name: str | None = None,
    description: str = "",
    compress: bool = True,
) -> Path:
    """Pack an adapter directory into a .mochi package.

    Args:
        adapter_dir: Directory containing trained adapter
        output_path: Output path for package (default: {name}.mochi or {name}.mochi.tar.gz)
        name: Package name (default: directory name)
        description: Package description
        compress: Create compressed archive (default: True)

    Returns:
        Path to created package
    """
    adapter_dir = Path(adapter_dir).resolve()

    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    # Determine package name
    package_name = name or adapter_dir.name
    if package_name.endswith(".mochi"):
        package_name = package_name[:-6]

    # Load existing config if present
    config_path = adapter_dir / "adapter_config.json"
    adapter_config_path = adapter_dir / "adapter" / "adapter_config.json"

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    elif adapter_config_path.exists():
        with open(adapter_config_path) as f:
            config = json.load(f)
    else:
        config = {}

    # Create manifest
    manifest = PackageManifest(
        name=package_name,
        version="1.0.0",
        base_model=config.get("base_model", "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"),
        adapter_type=config.get("type", "project"),
        description=description or config.get("description", ""),
        patterns=config.get("patterns", []),
        languages=config.get("languages", ["typescript"]),
        train_examples=config.get("train_examples", 0),
        final_loss=config.get("final_val_loss") or config.get("final_train_loss"),
    )

    # Determine output path
    if output_path is None:
        suffix = ".mochi.tar.gz" if compress else ".mochi"
        output_path = adapter_dir.parent / f"{package_name}{suffix}"
    output_path = Path(output_path)

    if compress:
        # Create tar.gz archive
        return _create_archive(adapter_dir, output_path, manifest, package_name)
    else:
        # Create directory package
        return _create_directory_package(adapter_dir, output_path, manifest)


def _create_archive(
    adapter_dir: Path,
    output_path: Path,
    manifest: PackageManifest,
    package_name: str,
) -> Path:
    """Create compressed .mochi.tar.gz archive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_dir = Path(tmpdir) / f"{package_name}.mochi"
        pkg_dir.mkdir()

        # Save manifest
        manifest.save(pkg_dir / "manifest.json")

        # Copy adapter files
        adapter_subdir = adapter_dir / "adapter"
        if adapter_subdir.exists():
            # Copy from adapter subdirectory
            for item in adapter_subdir.iterdir():
                if item.is_file():
                    shutil.copy2(item, pkg_dir / item.name)
        else:
            # Copy from root (weights directly in directory)
            for item in adapter_dir.iterdir():
                if item.is_file() and item.suffix in (".safetensors", ".json"):
                    shutil.copy2(item, pkg_dir / item.name)

        # Merge adapter_config.json from both locations
        # mlx-lm config (from adapter/) + mochi metadata (from root)
        merged_config = _merge_adapter_configs(adapter_dir)
        with open(pkg_dir / "adapter_config.json", "w") as f:
            json.dump(merged_config, f, indent=2)

        # Create archive
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(pkg_dir, arcname=f"{package_name}.mochi")

    return output_path


def _create_directory_package(
    adapter_dir: Path,
    output_path: Path,
    manifest: PackageManifest,
) -> Path:
    """Create uncompressed .mochi directory package."""
    output_path.mkdir(parents=True, exist_ok=True)

    # Save manifest
    manifest.save(output_path / "manifest.json")

    # Copy adapter files
    adapter_subdir = adapter_dir / "adapter"
    if adapter_subdir.exists():
        for item in adapter_subdir.iterdir():
            if item.is_file():
                shutil.copy2(item, output_path / item.name)
    else:
        for item in adapter_dir.iterdir():
            if item.is_file() and item.suffix in (".safetensors", ".json"):
                shutil.copy2(item, output_path / item.name)

    # Merge adapter_config.json from both locations
    # mlx-lm config (from adapter/) + mochi metadata (from root)
    merged_config = _merge_adapter_configs(adapter_dir)
    with open(output_path / "adapter_config.json", "w") as f:
        json.dump(merged_config, f, indent=2)

    return output_path


def _merge_adapter_configs(adapter_dir: Path) -> dict[str, Any]:
    """Merge adapter configs from mlx-lm output and mochi metadata.

    mlx-lm requires: fine_tune_type, lora_parameters, num_layers
    mochi uses: name, base_model, adapter_type, languages, etc.

    Args:
        adapter_dir: Adapter directory containing config files

    Returns:
        Merged config dictionary with all required fields
    """
    merged: dict[str, Any] = {}

    # Load mlx-lm config from adapter subdirectory (has num_layers, lora_parameters)
    mlx_config_path = adapter_dir / "adapter" / "adapter_config.json"
    if mlx_config_path.exists():
        with open(mlx_config_path) as f:
            mlx_config = json.load(f)
            merged.update(mlx_config)

    # Load mochi config from root (has name, adapter_type, base_model, etc.)
    mochi_config_path = adapter_dir / "adapter_config.json"
    if mochi_config_path.exists():
        with open(mochi_config_path) as f:
            mochi_config = json.load(f)
            # Only add mochi-specific fields, don't overwrite mlx-lm fields
            for key in ["name", "adapter_type", "base_adapter", "project_root",
                       "languages", "include_patterns", "exclude_patterns",
                       "version", "description", "metadata"]:
                if key in mochi_config and mochi_config[key] is not None:
                    merged[key] = mochi_config[key]
            # base_model from mochi config should match model from mlx config
            if "base_model" in mochi_config:
                merged["base_model"] = mochi_config["base_model"]

    return merged


def install_package(
    source: str | Path,
    target_dir: Path | None = None,
    name: str | None = None,
) -> Path:
    """Install a .mochi package.

    Args:
        source: Package source (local path, HTTP URL, or S3 URL)
        target_dir: Installation directory (default: ~/.mochi/adapters/)
        name: Override adapter name

    Returns:
        Path to installed adapter
    """
    target_dir = target_dir or DEFAULT_ADAPTERS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    source_str = str(source)

    # Determine source type and download if needed
    if source_str.startswith(("http://", "https://")):
        local_path = _download_http(source_str)
    elif source_str.startswith("s3://"):
        local_path = _download_s3(source_str)
    else:
        local_path = Path(source)

    if not local_path.exists():
        raise FileNotFoundError(f"Package not found: {local_path}")

    # Load package
    package = MochiPackage(local_path)
    manifest = package.manifest
    adapter_name = name or manifest.name

    # Determine final installation path
    install_path = target_dir / adapter_name

    # Remove existing if present
    if install_path.exists():
        shutil.rmtree(install_path)

    if package._is_archive:
        # Extract archive
        extracted = package.extract_to(target_dir)
        # Rename if needed
        if extracted.name != adapter_name:
            extracted.rename(install_path)
            extracted = install_path
        return extracted
    else:
        # Copy directory
        shutil.copytree(local_path, install_path)
        return install_path


def _download_http(url: str) -> Path:
    """Download package from HTTP URL."""
    cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Generate cache filename from URL
    parsed = urlparse(url)
    filename = Path(parsed.path).name
    cache_path = cache_dir / filename

    # Download if not cached
    if not cache_path.exists():
        print(f"Downloading {url}...")
        urlretrieve(url, cache_path)
        print(f"Downloaded to {cache_path}")

    return cache_path


def _download_s3(url: str) -> Path:
    """Download package from S3 URL."""
    try:
        import boto3
    except ImportError:
        raise ImportError("boto3 required for S3 downloads. Install with: pip install boto3")

    cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Parse S3 URL: s3://bucket/key
    parsed = urlparse(url)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    filename = Path(key).name
    cache_path = cache_dir / filename

    # Download if not cached
    if not cache_path.exists():
        print(f"Downloading from S3: {url}...")
        s3 = boto3.client("s3")
        s3.download_file(bucket, key, str(cache_path))
        print(f"Downloaded to {cache_path}")

    return cache_path


def list_installed_adapters(adapters_dir: Path | None = None) -> list[dict[str, Any]]:
    """List all installed adapters.

    Args:
        adapters_dir: Directory to scan (default: ~/.mochi/adapters/)

    Returns:
        List of adapter info dictionaries
    """
    adapters_dir = adapters_dir or DEFAULT_ADAPTERS_DIR

    if not adapters_dir.exists():
        return []

    adapters = []
    for item in adapters_dir.iterdir():
        if item.is_dir():
            try:
                package = MochiPackage(item)
                manifest = package.manifest
                adapters.append({
                    "name": manifest.name,
                    "path": str(item),
                    "type": manifest.adapter_type,
                    "base_model": manifest.base_model,
                    "version": manifest.version,
                    "description": manifest.description,
                })
            except (FileNotFoundError, json.JSONDecodeError):
                # Skip invalid packages
                continue

    return adapters


def get_default_adapter() -> Path | None:
    """Get the default adapter path.

    Looks for:
    1. $MOCHI_ADAPTER environment variable
    2. .mochi/adapters/default symlink or directory
    3. First installed adapter

    Returns:
        Path to default adapter or None
    """
    import os

    # Check environment variable
    env_adapter = os.environ.get("MOCHI_ADAPTER")
    if env_adapter:
        path = Path(env_adapter).expanduser()
        if path.exists():
            return path

    adapters_dir = DEFAULT_ADAPTERS_DIR

    # Check for default symlink/directory
    default_path = adapters_dir / "default"
    if default_path.exists():
        return default_path.resolve() if default_path.is_symlink() else default_path

    # Return first installed adapter
    adapters = list_installed_adapters(adapters_dir)
    if adapters:
        return Path(adapters[0]["path"])

    return None
