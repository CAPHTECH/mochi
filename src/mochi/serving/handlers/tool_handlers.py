"""Tool handlers for MCP Server.

Contains handlers for domain_query and complete_code tools.

Extracted from mcp_server.py to improve AI readability and maintainability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ..mcp_server import MCPServer


class HasEngine(Protocol):
    """Protocol for objects that provide an inference engine."""

    def _require_engine(self) -> Any:
        """Get initialized engine or raise RuntimeError."""
        ...

    def _make_error_response(self, message: str) -> dict[str, Any]:
        """Create error response."""
        ...

    def _make_success_response(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create success response."""
        ...


class ToolHandlers:
    """Mixin class providing tool handler methods.

    Provides handlers for:
    - domain_query: General code completion queries
    - complete_code: Code completion with prefix/suffix

    Usage:
        class MCPServer(ToolHandlers):
            ...
    """

    def get_tool_handlers(self: HasEngine) -> dict[str, Any]:
        """Get mapping of tool names to handler methods.

        Returns:
            Dict mapping tool name to handler callable
        """
        return {
            "domain_query": self._handle_domain_query,
            "complete_code": self._handle_complete_code,
        }

    def _handle_domain_query(self: HasEngine, args: dict[str, Any]) -> dict[str, Any]:
        """Handle domain_query tool call.

        Args:
            args: Tool arguments with instruction, input, file_path, etc.

        Returns:
            MCP response with generated code
        """
        try:
            engine = self._require_engine()
        except Exception as e:
            return self._make_error_response(f"Initialization error: {e}")

        instruction = args.get("instruction", "")
        input_code = args.get("input", "")
        file_path = args.get("file_path")
        max_tokens = args.get("max_tokens")
        temperature = args.get("temperature")
        use_lsp = args.get("use_lsp")

        try:
            response = engine.complete(
                instruction=instruction,
                input_code=input_code,
                file_path=file_path,
                max_tokens=max_tokens,
                temperature=temperature,
                use_lsp=use_lsp,
            )

            return self._make_success_response({
                "response": response,
                "adapter": str(engine.active_adapter),
            })

        except Exception as e:
            return self._make_error_response(f"Error: {e}")

    def _handle_complete_code(self: HasEngine, args: dict[str, Any]) -> dict[str, Any]:
        """Handle complete_code tool call.

        Args:
            args: Tool arguments with prefix, suffix, file_path, max_tokens

        Returns:
            MCP response with completion
        """
        try:
            engine = self._require_engine()
        except Exception as e:
            return self._make_error_response(f"Initialization error: {e}")

        prefix = args.get("prefix", "")
        suffix = args.get("suffix", "")
        file_path = args.get("file_path")
        max_tokens = args.get("max_tokens", 512)

        try:
            if suffix:
                input_code = f"{prefix}<FILL>{suffix}"
                instruction = "Fill in the code at <FILL> marker. Output only the code to insert, no explanations."
            else:
                input_code = prefix
                instruction = "Continue the code. Output only the code continuation, no explanations."

            response = engine.complete(
                instruction=instruction,
                input_code=input_code,
                file_path=file_path,
                max_tokens=max_tokens,
            )

            return self._make_success_response({"completion": response})

        except Exception as e:
            return self._make_error_response(f"Error: {e}")
