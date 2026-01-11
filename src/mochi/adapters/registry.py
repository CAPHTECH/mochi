"""Adapter registry for managing and discovering adapters.

The registry provides a central location for registering, discovering,
and loading adapters by name.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.exceptions import AdapterError, AdapterNotFoundError
from ..core.types import AdapterType

if TYPE_CHECKING:
    from .base_adapter import BaseAdapter
    from .project_adapter import ProjectAdapter

logger = logging.getLogger(__name__)

# Global registry instance
_registry: AdapterRegistry | None = None


class AdapterRegistry:
    """Central registry for managing adapters.

    The registry maintains a catalog of available adapters and their locations,
    enabling discovery and loading by name.

    Usage:
        registry = get_registry()

        # Register adapters
        registry.register(base_adapter)
        registry.register(project_adapter)

        # List available adapters
        for name, info in registry.list_adapters().items():
            print(f"{name}: {info['type']}")

        # Load adapter by name
        adapter = registry.load("my-project-adapter")
    """

    def __init__(self, adapters_dir: Path | str | None = None) -> None:
        """Initialize registry.

        Args:
            adapters_dir: Directory for storing/loading adapters
        """
        self.adapters_dir = Path(adapters_dir) if adapters_dir else Path("adapters")
        self._base_adapters: dict[str, BaseAdapter] = {}
        self._project_adapters: dict[str, ProjectAdapter] = {}
        self._catalog: dict[str, dict] = {}

    def register(
        self,
        adapter: BaseAdapter | ProjectAdapter,
        overwrite: bool = False,
    ) -> None:
        """Register an adapter.

        Args:
            adapter: Adapter to register
            overwrite: If True, overwrite existing registration
        """
        from .base_adapter import BaseAdapter
        from .project_adapter import ProjectAdapter

        name = adapter.name

        if not overwrite and name in self._catalog:
            raise AdapterError(
                f"Adapter already registered: {name}",
                {"existing_type": self._catalog[name]["type"]},
            )

        if isinstance(adapter, BaseAdapter):
            self._base_adapters[name] = adapter
            self._catalog[name] = {
                "type": AdapterType.BASE.value,
                "path": str(adapter.adapter_path) if adapter.adapter_path else None,
                "patterns": adapter.config.patterns,
            }
        elif isinstance(adapter, ProjectAdapter):
            self._project_adapters[name] = adapter
            self._catalog[name] = {
                "type": AdapterType.PROJECT.value,
                "path": str(adapter.adapter_path) if adapter.adapter_path else None,
                "base_adapter": adapter.config.base_adapter,
                "project_root": str(adapter.project_root) if adapter.project_root else None,
            }
        else:
            raise AdapterError(f"Unknown adapter type: {type(adapter)}")

        logger.info(f"Registered adapter: {name}")

    def unregister(self, name: str) -> bool:
        """Unregister an adapter.

        Args:
            name: Name of adapter to unregister

        Returns:
            True if adapter was found and unregistered
        """
        if name not in self._catalog:
            return False

        adapter_type = self._catalog[name]["type"]
        del self._catalog[name]

        if adapter_type == AdapterType.BASE.value:
            self._base_adapters.pop(name, None)
        else:
            self._project_adapters.pop(name, None)

        logger.info(f"Unregistered adapter: {name}")
        return True

    def get(
        self,
        name: str,
        lazy: bool = True,
    ) -> BaseAdapter | ProjectAdapter:
        """Get a registered adapter by name.

        Args:
            name: Adapter name
            lazy: If loading from disk, use lazy loading

        Returns:
            Adapter instance

        Raises:
            AdapterNotFoundError: If adapter not found
        """
        # Check in-memory adapters first
        if name in self._base_adapters:
            return self._base_adapters[name]
        if name in self._project_adapters:
            return self._project_adapters[name]

        # Try to load from catalog
        if name in self._catalog:
            return self._load_from_catalog(name, lazy=lazy)

        # Try to discover from disk
        adapter = self._discover_adapter(name, lazy=lazy)
        if adapter:
            return adapter

        raise AdapterNotFoundError(name, [str(self.adapters_dir)])

    def _load_from_catalog(
        self,
        name: str,
        lazy: bool = True,
    ) -> BaseAdapter | ProjectAdapter:
        """Load adapter from catalog entry."""
        from .base_adapter import BaseAdapter
        from .project_adapter import ProjectAdapter

        info = self._catalog[name]
        path = info.get("path")

        if not path:
            raise AdapterError(
                f"Adapter has no path: {name}",
                {"catalog_entry": info},
            )

        if info["type"] == AdapterType.BASE.value:
            adapter = BaseAdapter.load(path, lazy=lazy)
            self._base_adapters[name] = adapter
            return adapter
        else:
            # Load with base adapter if specified
            base_adapter = None
            if base_name := info.get("base_adapter"):
                base_adapter = self.get(base_name, lazy=lazy)

            adapter = ProjectAdapter.load(path, base_adapter=base_adapter, lazy=lazy)
            self._project_adapters[name] = adapter
            return adapter

    def _discover_adapter(
        self,
        name: str,
        lazy: bool = True,
    ) -> BaseAdapter | ProjectAdapter | None:
        """Try to discover and load adapter from disk."""
        from .base_adapter import BaseAdapter
        from .project_adapter import ProjectAdapter

        # Check adapters directory
        adapter_path = self.adapters_dir / name
        if not adapter_path.exists():
            return None

        config_path = adapter_path / "adapter_config.json"
        if not config_path.exists():
            return None

        # Read config to determine type
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        adapter_type = config_data.get("adapter_type", "base")

        if adapter_type == AdapterType.BASE.value:
            adapter = BaseAdapter.load(adapter_path, lazy=lazy)
            self.register(adapter)
            return adapter
        else:
            # Load with base adapter if specified
            base_adapter = None
            if base_name := config_data.get("base_adapter"):
                try:
                    base_adapter = self.get(base_name, lazy=lazy)
                except AdapterNotFoundError:
                    logger.warning(f"Base adapter not found: {base_name}")

            adapter = ProjectAdapter.load(adapter_path, base_adapter=base_adapter, lazy=lazy)
            self.register(adapter)
            return adapter

    def list_adapters(
        self,
        adapter_type: AdapterType | None = None,
    ) -> dict[str, dict]:
        """List registered adapters.

        Args:
            adapter_type: Filter by adapter type (optional)

        Returns:
            Dictionary mapping adapter names to their info
        """
        if adapter_type is None:
            return dict(self._catalog)

        return {
            name: info
            for name, info in self._catalog.items()
            if info["type"] == adapter_type.value
        }

    def list_base_adapters(self) -> dict[str, dict]:
        """List registered base adapters."""
        return self.list_adapters(AdapterType.BASE)

    def list_project_adapters(self) -> dict[str, dict]:
        """List registered project adapters."""
        return self.list_adapters(AdapterType.PROJECT)

    def discover_all(self, lazy: bool = True) -> int:
        """Discover all adapters in the adapters directory.

        Args:
            lazy: Use lazy loading for discovered adapters

        Returns:
            Number of adapters discovered
        """
        if not self.adapters_dir.exists():
            return 0

        count = 0
        for path in self.adapters_dir.iterdir():
            if not path.is_dir():
                continue

            config_path = path / "adapter_config.json"
            if not config_path.exists():
                continue

            try:
                self._discover_adapter(path.name, lazy=lazy)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to discover adapter at {path}: {e}")

        logger.info(f"Discovered {count} adapters")
        return count

    def save_catalog(self, path: Path | str | None = None) -> None:
        """Save the adapter catalog to disk.

        Args:
            path: Output path (defaults to adapters_dir/catalog.json)
        """
        path = Path(path) if path else self.adapters_dir / "catalog.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._catalog, f, indent=2)

        logger.info(f"Saved adapter catalog to {path}")

    def load_catalog(self, path: Path | str | None = None) -> int:
        """Load the adapter catalog from disk.

        Args:
            path: Catalog path (defaults to adapters_dir/catalog.json)

        Returns:
            Number of adapters loaded into catalog
        """
        path = Path(path) if path else self.adapters_dir / "catalog.json"

        if not path.exists():
            logger.debug(f"Catalog not found at {path}")
            return 0

        with open(path, "r", encoding="utf-8") as f:
            self._catalog = json.load(f)

        logger.info(f"Loaded {len(self._catalog)} adapters from catalog")
        return len(self._catalog)

    def clear(self) -> None:
        """Clear all registered adapters."""
        self._base_adapters.clear()
        self._project_adapters.clear()
        self._catalog.clear()


def get_registry(adapters_dir: Path | str | None = None) -> AdapterRegistry:
    """Get the global adapter registry.

    Args:
        adapters_dir: Directory for adapters (only used on first call)

    Returns:
        Global AdapterRegistry instance
    """
    global _registry

    if _registry is None:
        _registry = AdapterRegistry(adapters_dir)

    return _registry


def reset_registry() -> None:
    """Reset the global registry (mainly for testing)."""
    global _registry
    _registry = None
