"""Adapter management for mochi library.

This module provides the core adapter classes:
- BaseAdapter: Pre-trained adapters for common patterns
- ProjectAdapter: Project-specific fine-tuned adapters
- AdapterStack: Runtime composition of multiple adapters
"""

from .base_adapter import BaseAdapter
from .project_adapter import ProjectAdapter
from .adapter_stack import AdapterStack
from .registry import AdapterRegistry, get_registry

__all__ = [
    "BaseAdapter",
    "ProjectAdapter",
    "AdapterStack",
    "AdapterRegistry",
    "get_registry",
]
