"""Mochi MCP Server - Expose fine-tuned SLM via MCP protocol."""

from mochi.mcp.server import MochiMCPServer

# Lazy import to avoid torch dependency when using MLX
def get_inference_engine():
    """Get the PyTorch-based InferenceEngine (requires torch)."""
    from mochi.mcp.inference import InferenceEngine
    return InferenceEngine

__all__ = ["MochiMCPServer", "get_inference_engine"]
