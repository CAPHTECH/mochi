"""Mochi - Domain-specific SLM generator for software projects.

mochi provides Base Adapters (common patterns) and Project Adapters
(project-specific patterns) for enhanced code completion.

Quick Start:
    from mochi.adapters import BaseAdapter, ProjectAdapter, AdapterStack
    from mochi.inference import InferenceEngine

    # Load adapters
    base = BaseAdapter.load("output/base-adapter/")
    project = ProjectAdapter.load("output/project-adapter/", base_adapter=base)

    # Create stack
    stack = AdapterStack([(base, 0.3), (project, 0.7)])

    # Run inference
    engine = InferenceEngine(adapter_stack=stack)
    result = engine.complete(
        instruction="Fill in the code",
        input_code="const users = await db.",
    )
"""

__version__ = "0.2.0"


# Lazy imports to avoid heavy dependencies at module load
def get_base_adapter():
    """Get BaseAdapter class."""
    from mochi.adapters.base_adapter import BaseAdapter
    return BaseAdapter


def get_project_adapter():
    """Get ProjectAdapter class."""
    from mochi.adapters.project_adapter import ProjectAdapter
    return ProjectAdapter


def get_adapter_stack():
    """Get AdapterStack class."""
    from mochi.adapters.adapter_stack import AdapterStack
    return AdapterStack


def get_inference_engine():
    """Get InferenceEngine class."""
    from mochi.inference.engine import InferenceEngine
    return InferenceEngine


def get_mcp_server():
    """Get MCPServer class."""
    from mochi.serving.mcp_server import MCPServer
    return MCPServer


# Legacy compatibility
def get_legacy_mcp_server():
    """Get legacy MochiMCPServer class (deprecated)."""
    from mochi.mcp.server import MochiMCPServer
    return MochiMCPServer
