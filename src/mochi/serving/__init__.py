"""Serving module for mochi library.

Provides MCP server for Claude Code integration.
"""

from .mcp_server import MCPServer, start_server

__all__ = [
    "MCPServer",
    "start_server",
]
