"""Packaging module for mochi adapters.

Provides functionality to:
- Pack trained adapters into distributable .mochi packages
- Install .mochi packages from URLs or local files
- Manage installed adapters in ~/.mochi/adapters/
"""

from .package import (
    DEFAULT_ADAPTERS_DIR,
    DEFAULT_MOCHI_DIR,
    MochiPackage,
    PackageManifest,
    get_default_adapter,
    install_package,
    list_installed_adapters,
    pack_adapter,
)

__all__ = [
    "DEFAULT_ADAPTERS_DIR",
    "DEFAULT_MOCHI_DIR",
    "MochiPackage",
    "PackageManifest",
    "get_default_adapter",
    "install_package",
    "list_installed_adapters",
    "pack_adapter",
]
