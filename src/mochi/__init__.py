"""Mochi - Domain-specific SLM generator for software projects."""

__version__ = "0.1.0"

# Lazy imports to avoid heavy dependencies at module load
def get_mcp_server():
    """Get MCP server class."""
    from mochi.mcp.server import MochiMCPServer
    return MochiMCPServer

def get_inference_engine():
    """Get inference engine class."""
    from mochi.mcp.inference import InferenceEngine
    return InferenceEngine
