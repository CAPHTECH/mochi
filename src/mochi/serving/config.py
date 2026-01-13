"""MCP Server configuration classes.

Contains configuration dataclasses for the MCP server.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ToolDefinition:
    """MCP Tool definition.

    Defines a tool that can be called via the MCP protocol.

    Attributes:
        name: Tool identifier used in tools/call requests
        description: Human-readable description shown to clients
        input_schema: JSON Schema defining expected input parameters
    """

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class MCPServerConfig:
    """MCP Server configuration.

    Configures the MCP server behavior including model settings,
    adapter paths, and runtime parameters.

    Attributes:
        backend: Inference backend ("mlx" for Apple Silicon, "pytorch" for CUDA)
        base_model: HuggingFace model identifier
        base_adapter_path: Path to base adapter directory
        project_adapter_path: Path to project-specific adapter directory
        base_weight: Weight for base adapter in stack (0.0-1.0)
        project_weight: Weight for project adapter in stack (0.0-1.0)
        timeout_seconds: Request timeout in seconds
        max_tokens: Maximum tokens to generate per request
        temperature: Sampling temperature (0.0-1.0)
        top_p: Top-p sampling parameter
        patterns_dir: Directory containing pattern markdown files
        conventions_dir: Directory containing convention markdown files
        use_lsp_context: Enable LSP context extraction
        lsp_context_lines: Maximum lines of LSP context to include
        project_root: Project root for LSP initialization
        lsp_language: Language server type (typescript, python, etc.)
        schema_path: Path to schema.yaml for database context
    """

    # Backend selection: "mlx" (recommended for Apple Silicon) or "pytorch"
    backend: str = "mlx"

    # Model settings
    base_model: str = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"

    # Adapter paths
    base_adapter_path: Path | None = None
    project_adapter_path: Path | None = None

    # Adapter weights for stack
    base_weight: float = 0.3
    project_weight: float = 0.7

    # Runtime settings
    timeout_seconds: float = 30.0
    max_tokens: int = 2048
    temperature: float = 0.1
    top_p: float = 0.9

    # Resource directories
    patterns_dir: Path | None = None
    conventions_dir: Path | None = None

    # LSP settings
    use_lsp_context: bool = True
    lsp_context_lines: int = 50

    # Project root for LSP context extraction
    # If None, LSP context will not be available
    project_root: Path | None = None
    lsp_language: str = "typescript"
    schema_path: Path | None = None
