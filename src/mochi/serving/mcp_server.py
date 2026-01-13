"""MCP Server for mochi library.

Provides MCP (Model Context Protocol) server for Claude Code integration.
Supports both the new adapter architecture and legacy inference engines.

Law compliance:
- L-mcp-compliance: JSON-RPC 2.0 protocol
- L-adapter-required: Initialization check
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.async_utils import run_sync
from ..core.exceptions import AdapterError, InferenceError
from .config import MCPServerConfig, ToolDefinition
from .handlers import ToolHandlers
from .tool_definitions import TOOLS

if TYPE_CHECKING:
    from ..adapters.adapter_stack import AdapterStack
    from ..adapters.base_adapter import BaseAdapter
    from ..adapters.project_adapter import ProjectAdapter
    from ..inference.engine import InferenceEngine

logger = logging.getLogger(__name__)


class MCPServer(ToolHandlers):
    """MCP Server for mochi inference.

    Implements JSON-RPC 2.0 over stdio for Claude Code integration.
    Inherits tool handlers from ToolHandlers mixin.

    Usage:
        # With adapter stack
        server = MCPServer(
            config=MCPServerConfig(
                base_adapter_path=Path("output/base/"),
                project_adapter_path=Path("output/project/"),
            )
        )
        server.run_stdio()

        # Or with pre-configured engine
        engine = InferenceEngine(adapter_stack=stack)
        server = MCPServer(inference_engine=engine)
        server.run_stdio()
    """

    # Tool definitions imported from tool_definitions.py

    def __init__(
        self,
        config: MCPServerConfig | None = None,
        inference_engine: InferenceEngine | None = None,
    ) -> None:
        """Initialize MCP server.

        Args:
            config: Server configuration (used if inference_engine not provided)
            inference_engine: Pre-configured inference engine
        """
        self.config = config or MCPServerConfig()
        self._engine = inference_engine
        self._context_extractor = None  # ContextExtractor for LSP context
        self._initialized = False

        # Resource storage
        self._patterns: dict[str, str] = {}
        self._conventions: dict[str, str] = {}

    def initialize(self) -> None:
        """Initialize the server and load adapters.

        Raises:
            RuntimeError: If initialization fails
        """
        if self._initialized:
            return

        logger.info("Initializing MCP server...")

        if self._engine is None:
            self._create_inference_engine()

        # Load patterns if directory specified
        if self.config.patterns_dir:
            self._load_patterns(self.config.patterns_dir)

        # Load conventions if directory specified
        if self.config.conventions_dir:
            self._load_conventions(self.config.conventions_dir)

        self._initialized = True
        logger.info("MCP server initialized.")

    def _create_inference_engine(self) -> None:
        """Create inference engine from config."""
        from ..adapters.adapter_stack import AdapterStack
        from ..adapters.base_adapter import BaseAdapter
        from ..adapters.project_adapter import ProjectAdapter
        from ..core.types import InferenceConfig
        from ..inference.engine import InferenceEngine

        adapters: list[tuple[BaseAdapter | ProjectAdapter, float]] = []

        # Load base adapter if specified
        base_adapter = None
        if self.config.base_adapter_path:
            logger.info(f"Loading base adapter from: {self.config.base_adapter_path}")
            base_adapter = BaseAdapter.load(self.config.base_adapter_path, lazy=False)
            adapters.append((base_adapter, self.config.base_weight))

        # Load project adapter if specified
        if self.config.project_adapter_path:
            logger.info(f"Loading project adapter from: {self.config.project_adapter_path}")
            project_adapter = ProjectAdapter.load(
                self.config.project_adapter_path,
                base_adapter=base_adapter,
                lazy=False,
            )
            adapters.append((project_adapter, self.config.project_weight))

        if not adapters:
            # Create a dummy base adapter with just the base model
            logger.warning("No adapters specified, using base model only")
            base_adapter = BaseAdapter(
                name="base-model",
                adapter_path=None,  # type: ignore
                base_model=self.config.base_model,
            )
            adapters.append((base_adapter, 1.0))

        # Create adapter stack or use single adapter
        inference_config = InferenceConfig(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            use_lsp_context=self.config.use_lsp_context,
            lsp_context_lines=self.config.lsp_context_lines,
        )

        # Initialize ContextExtractor if project_root is specified
        context_extractor = None
        if self.config.project_root and self.config.use_lsp_context:
            try:
                context_extractor = self._create_context_extractor()
                logger.info(f"ContextExtractor initialized for: {self.config.project_root}")
            except Exception as e:
                logger.warning(f"Failed to initialize ContextExtractor: {e}")
                # Continue without LSP context

        if len(adapters) > 1:
            stack = AdapterStack(adapters)
            self._engine = InferenceEngine(
                adapter_stack=stack,
                context_extractor=context_extractor,
                config=inference_config,
            )
        else:
            self._engine = InferenceEngine(
                adapter=adapters[0][0],
                context_extractor=context_extractor,
                config=inference_config,
            )

        # Store for cleanup
        self._context_extractor = context_extractor

    def _create_context_extractor(self):
        """Create ContextExtractor for LSP context.

        Returns:
            ContextExtractor instance or None if creation fails
        """
        from ..lsp.client import LSPClient
        from ..lsp.context_extractor import ContextExtractor

        try:
            # Create LSP client
            lsp_client = LSPClient(
                language=self.config.lsp_language,
                project_root=self.config.project_root,
            )

            # Start LSP client (async)
            run_sync(lsp_client.start())

            # Create extractor
            return ContextExtractor(
                lsp_client=lsp_client,
                schema_path=self.config.schema_path,
            )

        except Exception as e:
            logger.warning(f"ContextExtractor creation failed: {e}")
            return None

    def _ensure_initialized(self) -> None:
        """Ensure server is initialized (lazy initialization)."""
        if not self._initialized:
            self.initialize()

    def _require_engine(self) -> "InferenceEngine":
        """Ensure engine is initialized and return it.

        Returns:
            InferenceEngine instance

        Raises:
            RuntimeError: If initialization fails or engine is not available
        """
        self._ensure_initialized()
        if not self._engine:
            raise RuntimeError("Inference engine not initialized.")
        return self._engine

    def _make_error_response(self, message: str) -> dict[str, Any]:
        """Create a standard error response.

        Args:
            message: Error message

        Returns:
            MCP error response dict
        """
        return {
            "content": [{"type": "text", "text": message}],
            "isError": True,
        }

    def _make_success_response(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create a standard success response.

        Args:
            data: Response data to serialize

        Returns:
            MCP success response dict
        """
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(data, indent=2, ensure_ascii=False),
                }
            ]
        }

    def _load_patterns(self, patterns_dir: Path) -> None:
        """Load patterns from directory."""
        if not patterns_dir.exists():
            return

        for file in patterns_dir.glob("*.md"):
            name = file.stem
            self._patterns[name] = file.read_text()

    def _load_conventions(self, conventions_dir: Path) -> None:
        """Load conventions from directory."""
        if not conventions_dir.exists():
            return

        for file in conventions_dir.glob("*.md"):
            name = file.stem
            self._conventions[name] = file.read_text()

    # === JSON-RPC handlers ===

    def handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle initialize request."""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
                "resources": {},
            },
            "serverInfo": {
                "name": "mochi",
                "version": "0.2.0",
            },
        }

    def handle_tools_list(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tools/list request."""
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema,
                }
                for tool in TOOLS
            ]
        }

    def handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        # Get handlers from mixin (see handlers/tool_handlers.py)
        handlers = self.get_tool_handlers()
        handler = handlers.get(tool_name)
        if handler:
            return handler(arguments)
        else:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Unknown tool: {tool_name}",
                    }
                ],
                "isError": True,
            }

    # Tool handlers are inherited from ToolHandlers mixin
    # See: handlers/tool_handlers.py

    def handle_resources_list(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle resources/list request."""
        resources = []

        # Add pattern resources
        for name in self._patterns:
            resources.append({
                "uri": f"mochi://patterns/{name}",
                "name": f"Pattern: {name}",
                "mimeType": "text/markdown",
                "description": f"Code pattern: {name}",
            })

        # Add convention resources
        for name in self._conventions:
            resources.append({
                "uri": f"mochi://conventions/{name}",
                "name": f"Convention: {name}",
                "mimeType": "text/markdown",
                "description": f"Coding convention: {name}",
            })

        # Add stats resource
        resources.append({
            "uri": "mochi://stats",
            "name": "Server Statistics",
            "mimeType": "application/json",
            "description": "Server and adapter info",
        })

        return {"resources": resources}

    def handle_resources_read(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle resources/read request."""
        uri = params.get("uri", "")

        if uri.startswith("mochi://patterns/"):
            name = uri.replace("mochi://patterns/", "")
            if name in self._patterns:
                return {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "text/markdown",
                            "text": self._patterns[name],
                        }
                    ]
                }
            else:
                return {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "text/plain",
                            "text": f"Pattern not found: {name}",
                        }
                    ]
                }

        elif uri.startswith("mochi://conventions/"):
            name = uri.replace("mochi://conventions/", "")
            if name in self._conventions:
                return {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "text/markdown",
                            "text": self._conventions[name],
                        }
                    ]
                }
            else:
                return {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "text/plain",
                            "text": f"Convention not found: {name}",
                        }
                    ]
                }

        elif uri == "mochi://stats":
            stats = {
                "backend": self.config.backend,
                "base_model": self.config.base_model,
                "base_adapter": str(self.config.base_adapter_path) if self.config.base_adapter_path else None,
                "project_adapter": str(self.config.project_adapter_path) if self.config.project_adapter_path else None,
                "initialized": self._initialized,
                "engine": str(self._engine) if self._engine else None,
                "patterns_count": len(self._patterns),
                "conventions_count": len(self._conventions),
            }
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(stats, indent=2, ensure_ascii=False),
                    }
                ]
            }

        else:
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "text/plain",
                        "text": f"Unknown resource: {uri}",
                    }
                ]
            }

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle a JSON-RPC request.

        Args:
            request: JSON-RPC 2.0 request

        Returns:
            JSON-RPC 2.0 response
        """
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")

        handlers = {
            "initialize": self.handle_initialize,
            "tools/list": self.handle_tools_list,
            "tools/call": self.handle_tools_call,
            "resources/list": self.handle_resources_list,
            "resources/read": self.handle_resources_read,
        }

        handler = handlers.get(method)
        if handler:
            try:
                result = handler(params)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": result,
                }
            except Exception as e:
                logger.exception(f"Handler error for {method}")
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32000,
                        "message": str(e),
                    },
                }
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            }

    def run_stdio(self) -> None:
        """Run server using stdio transport.

        Model loading is deferred to first tool call for faster MCP initialization.
        """
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break

                request = json.loads(line)
                response = self.handle_request(request)
                sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                sys.stdout.flush()

            except json.JSONDecodeError:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": "Parse error",
                    },
                }
                sys.stdout.write(json.dumps(error_response, ensure_ascii=False) + "\n")
                sys.stdout.flush()
            except KeyboardInterrupt:
                break

    def shutdown(self) -> None:
        """Shutdown server and cleanup resources."""
        # Cleanup LSP client if present
        if self._context_extractor:
            try:
                lsp_client = self._context_extractor.lsp
                if lsp_client:
                    run_sync(lsp_client.stop())
            except Exception as e:
                logger.debug(f"LSP client cleanup error: {e}")

        self._context_extractor = None
        self._initialized = False
        self._engine = None


def start_server(
    base_adapter: Path | str | None = None,
    project_adapter: Path | str | None = None,
    base_model: str = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
    base_weight: float = 0.3,
    project_weight: float = 0.7,
    preset: str | None = None,
    project_root: Path | str | None = None,
    lsp_language: str = "typescript",
    schema_path: Path | str | None = None,
) -> None:
    """Start MCP server with specified adapters.

    Args:
        base_adapter: Path to base adapter directory
        project_adapter: Path to project adapter directory
        base_model: Base model name
        base_weight: Weight for base adapter
        project_weight: Weight for project adapter
        preset: Preset configuration name (qwen3-coder, qwen3-coder-base, gpt-oss)
        project_root: Project root for LSP context extraction
        lsp_language: Language for LSP (typescript, python, etc.)
        schema_path: Path to schema.yaml for DB schema context
    """
    # Apply preset if specified
    if preset:
        try:
            from ..mcp.inference_mlx import PRESETS
            if preset in PRESETS:
                preset_config = PRESETS[preset]
                base_model = preset_config.get("model_path", base_model)
                default_adapter = preset_config.get("default_adapter")
                if default_adapter and not project_adapter:
                    project_adapter = default_adapter
                logger.info(f"Applied preset '{preset}': model={base_model}")
            else:
                logger.warning(f"Unknown preset: {preset}")
        except ImportError:
            logger.warning("Could not load PRESETS from inference_mlx")

    # Convert string paths to Path objects
    base_adapter_path = Path(base_adapter) if isinstance(base_adapter, str) else base_adapter
    project_adapter_path = Path(project_adapter) if isinstance(project_adapter, str) else project_adapter
    project_root_path = Path(project_root) if isinstance(project_root, str) else project_root
    schema_path_obj = Path(schema_path) if isinstance(schema_path, str) else schema_path

    config = MCPServerConfig(
        base_model=base_model,
        base_adapter_path=base_adapter_path,
        project_adapter_path=project_adapter_path,
        base_weight=base_weight,
        project_weight=project_weight,
        project_root=project_root_path,
        lsp_language=lsp_language,
        schema_path=schema_path_obj,
    )

    server = MCPServer(config=config)
    try:
        server.run_stdio()
    finally:
        server.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mochi MCP Server")
    parser.add_argument("--base", type=Path, help="Base adapter directory")
    parser.add_argument("--project", type=Path, help="Project adapter directory")
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
        help="Base model name",
    )
    parser.add_argument(
        "--preset",
        choices=["qwen3-coder", "qwen3-coder-base", "gpt-oss"],
        help="Use preset configuration",
    )
    parser.add_argument("--base-weight", type=float, default=0.3, help="Base adapter weight")
    parser.add_argument("--project-weight", type=float, default=0.7, help="Project adapter weight")
    parser.add_argument("--project-root", type=Path, help="Project root for LSP context extraction")
    parser.add_argument("--lsp-language", default="typescript", help="Language for LSP (typescript, python, etc.)")
    parser.add_argument("--schema", type=Path, help="Path to schema.yaml for DB context")

    args = parser.parse_args()

    start_server(
        base_adapter=args.base,
        project_adapter=args.project,
        base_model=args.model,
        base_weight=args.base_weight,
        project_weight=args.project_weight,
        preset=args.preset,
        project_root=args.project_root,
        lsp_language=args.lsp_language,
        schema_path=args.schema,
    )
