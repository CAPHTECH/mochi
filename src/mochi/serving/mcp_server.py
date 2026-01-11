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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.exceptions import AdapterError, InferenceError

if TYPE_CHECKING:
    from ..adapters.adapter_stack import AdapterStack
    from ..adapters.base_adapter import BaseAdapter
    from ..adapters.project_adapter import ProjectAdapter
    from ..inference.engine import InferenceEngine

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """MCP Tool definition."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class MCPServerConfig:
    """MCP Server configuration."""

    # Backend selection: "mlx" (recommended for Apple Silicon) or "pytorch"
    backend: str = "mlx"

    # Model settings
    base_model: str = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit"

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


class MCPServer:
    """MCP Server for mochi inference.

    Implements JSON-RPC 2.0 over stdio for Claude Code integration.

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

    # Tool definitions
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
                        "description": "Maximum tokens to generate (default: 256)",
                    },
                },
                "required": ["prefix"],
            },
        ),
        ToolDefinition(
            name="suggest_pattern",
            description=(
                "Generate code pattern suggestions based on goal description. "
                "Returns pattern examples based on learned conventions."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "context": {
                        "type": "string",
                        "description": "Current code context (imports, surrounding code)",
                    },
                    "goal": {
                        "type": "string",
                        "description": "What you want to implement",
                    },
                },
                "required": ["goal"],
            },
        ),
        ToolDefinition(
            name="generate_diff",
            description=(
                "Generate a unified diff for a code change. "
                "Outputs only the changed parts in unified diff format."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "original_code": {
                        "type": "string",
                        "description": "The original code to modify",
                    },
                    "change_description": {
                        "type": "string",
                        "description": "Description of the desired change",
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language (default: typescript)",
                    },
                },
                "required": ["original_code", "change_description"],
            },
        ),
    ]

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

        if len(adapters) > 1:
            stack = AdapterStack(adapters)
            self._engine = InferenceEngine(
                adapter_stack=stack,
                config=inference_config,
            )
        else:
            self._engine = InferenceEngine(
                adapter=adapters[0][0],
                config=inference_config,
            )

    def _ensure_initialized(self) -> None:
        """Ensure server is initialized (lazy initialization)."""
        if not self._initialized:
            self.initialize()

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
                for tool in self.TOOLS
            ]
        }

    def handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        handlers = {
            "domain_query": self._handle_domain_query,
            "complete_code": self._handle_complete_code,
            "suggest_pattern": self._handle_suggest_pattern,
            "generate_diff": self._handle_generate_diff,
        }

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

    def _handle_domain_query(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle domain_query tool call."""
        try:
            self._ensure_initialized()
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Initialization error: {e}"}],
                "isError": True,
            }

        if not self._engine:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Inference engine not initialized.",
                    }
                ],
                "isError": True,
            }

        instruction = args.get("instruction", "")
        input_code = args.get("input", "")
        file_path = args.get("file_path")
        max_tokens = args.get("max_tokens")
        temperature = args.get("temperature")
        use_lsp = args.get("use_lsp")

        try:
            response = self._engine.complete(
                instruction=instruction,
                input_code=input_code,
                file_path=file_path,
                max_tokens=max_tokens,
                temperature=temperature,
                use_lsp=use_lsp,
            )

            response_data = {
                "response": response,
                "adapter": str(self._engine.active_adapter),
            }

            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(response_data, indent=2, ensure_ascii=False),
                    }
                ]
            }

        except InferenceError as e:
            return {
                "content": [{"type": "text", "text": f"Inference error: {e}"}],
                "isError": True,
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "isError": True,
            }

    def _handle_complete_code(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle complete_code tool call."""
        try:
            self._ensure_initialized()
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Initialization error: {e}"}],
                "isError": True,
            }

        if not self._engine:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Inference engine not initialized.",
                    }
                ],
                "isError": True,
            }

        prefix = args.get("prefix", "")
        suffix = args.get("suffix", "")
        file_path = args.get("file_path")
        max_tokens = args.get("max_tokens", 256)

        try:
            # Build fill-in-the-middle style prompt
            if suffix:
                input_code = f"{prefix}<FILL>{suffix}"
                instruction = "Fill in the code at <FILL> marker"
            else:
                input_code = prefix
                instruction = "Continue the code"

            response = self._engine.complete(
                instruction=instruction,
                input_code=input_code,
                file_path=file_path,
                max_tokens=max_tokens,
            )

            response_data = {
                "completion": response,
            }

            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(response_data, indent=2, ensure_ascii=False),
                    }
                ]
            }

        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "isError": True,
            }

    def _handle_suggest_pattern(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle suggest_pattern tool call."""
        try:
            self._ensure_initialized()
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Initialization error: {e}"}],
                "isError": True,
            }

        if not self._engine:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Inference engine not initialized.",
                    }
                ],
                "isError": True,
            }

        context = args.get("context", "")
        goal = args.get("goal", "")

        try:
            instruction = f"Suggest code patterns for: {goal}"
            input_code = f"// Context:\n{context}\n\n// Goal: {goal}\n// Pattern:"

            response = self._engine.complete(
                instruction=instruction,
                input_code=input_code,
                max_tokens=1024,
            )

            response_data = {
                "patterns": [
                    {
                        "name": "Generated Pattern",
                        "description": goal,
                        "example": response,
                    }
                ],
            }

            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(response_data, indent=2, ensure_ascii=False),
                    }
                ]
            }

        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "isError": True,
            }

    def _handle_generate_diff(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle generate_diff tool call."""
        try:
            self._ensure_initialized()
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Initialization error: {e}"}],
                "isError": True,
            }

        if not self._engine:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Inference engine not initialized.",
                    }
                ],
                "isError": True,
            }

        original_code = args.get("original_code", "")
        change_description = args.get("change_description", "")
        language = args.get("language", "typescript")

        if not original_code or not change_description:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Both 'original_code' and 'change_description' are required.",
                    }
                ],
                "isError": True,
            }

        try:
            instruction = (
                f"Generate a unified diff for the following {language} code change. "
                f"Change: {change_description}"
            )
            input_code = f"```{language}\n{original_code}\n```\n\nGenerate unified diff:"

            response = self._engine.complete(
                instruction=instruction,
                input_code=input_code,
                max_tokens=1024,
            )

            response_data = {
                "diff": response,
            }

            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(response_data, indent=2, ensure_ascii=False),
                    }
                ]
            }

        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "isError": True,
            }

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
        self._initialized = False
        self._engine = None


def start_server(
    base_adapter: Path | None = None,
    project_adapter: Path | None = None,
    base_model: str = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    base_weight: float = 0.3,
    project_weight: float = 0.7,
) -> None:
    """Start MCP server with specified adapters.

    Args:
        base_adapter: Path to base adapter directory
        project_adapter: Path to project adapter directory
        base_model: Base model name
        base_weight: Weight for base adapter
        project_weight: Weight for project adapter
    """
    config = MCPServerConfig(
        base_model=base_model,
        base_adapter_path=base_adapter,
        project_adapter_path=project_adapter,
        base_weight=base_weight,
        project_weight=project_weight,
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
        default="mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
        help="Base model name",
    )
    parser.add_argument("--base-weight", type=float, default=0.3, help="Base adapter weight")
    parser.add_argument("--project-weight", type=float, default=0.7, help="Project adapter weight")

    args = parser.parse_args()

    start_server(
        base_adapter=args.base,
        project_adapter=args.project,
        base_model=args.model,
        base_weight=args.base_weight,
        project_weight=args.project_weight,
    )
