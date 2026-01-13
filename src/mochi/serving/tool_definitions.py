"""MCP Tool definitions for mochi server.

Contains the list of tools available via the MCP protocol.
Each tool has a name, description, and JSON Schema for input validation.
"""

from __future__ import annotations

from .config import ToolDefinition

# MCP Tool definitions
TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="domain_query",
        description=(
            "Query the domain-specific model for code completion. "
            "Use code-completion style instructions like 'Fill in the code', "
            "'Implement the following', 'Write the implementation'. "
            "Provide code context in the 'input' field. "
            "Model learns PATTERNS and STYLE - verify exact APIs against source."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": (
                        "Code completion instruction. Use: 'Fill in the code', "
                        "'Implement the following based on the context'"
                    ),
                },
                "input": {
                    "type": "string",
                    "description": (
                        "Code context with file path comment. "
                        "Format: '// File: path/to/file.ts\\n<code before completion point>'"
                    ),
                },
                "file_path": {
                    "type": "string",
                    "description": "File path for LSP context extraction",
                },
                "max_tokens": {
                    "type": "number",
                    "description": "Maximum tokens to generate (default: 2048)",
                },
                "temperature": {
                    "type": "number",
                    "description": "Sampling temperature 0.0-1.0 (default: 0.1)",
                },
                "use_lsp": {
                    "type": "boolean",
                    "description": "Use LSP context if available (default: true)",
                },
            },
            "required": ["instruction"],
        },
    ),
    ToolDefinition(
        name="complete_code",
        description=(
            "Complete code using trained adapter patterns. "
            "Provide code prefix (and optional suffix for fill-in-the-middle). "
            "Model generates continuation matching learned patterns."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "prefix": {
                    "type": "string",
                    "description": "Code before cursor position",
                },
                "suffix": {
                    "type": "string",
                    "description": "Code after cursor position (for fill-in-the-middle)",
                },
                "file_path": {
                    "type": "string",
                    "description": "File path for context",
                },
                "max_tokens": {
                    "type": "number",
                    "description": "Maximum tokens to generate (default: 512)",
                },
            },
            "required": ["prefix"],
        },
    ),
]
