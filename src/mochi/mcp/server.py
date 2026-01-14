"""Mochi MCP Server implementation.

Law compliance:
- L-mcp-compliance: JSON-RPC 2.0 protocol
- L-adapter-required: Initialization check

Supports both PyTorch and MLX backends.
MLX is recommended for Apple Silicon (much faster).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from mochi.validation.output_validator import OutputValidator, ValidationResult


class InferenceEngineProtocol(Protocol):
    """Protocol for inference engines (PyTorch or MLX)."""

    @property
    def is_loaded(self) -> bool: ...
    def load(self) -> None: ...
    def generate(
        self,
        instruction: str,
        input_text: str = "",
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        top_p: float = 0.5,
    ) -> Any: ...
    def complete(
        self,
        prefix: str,
        suffix: str = "",
        max_new_tokens: int = 256,
        num_alternatives: int = 3,
    ) -> list[str]: ...
    def unload(self) -> None: ...


@dataclass
class ToolDefinition:
    """MCP Tool definition."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ResourceDefinition:
    """MCP Resource definition."""

    uri: str
    name: str
    mime_type: str
    description: str


@dataclass
class ServerConfig:
    """Server configuration."""

    # Backend selection: "mlx" (recommended for Apple Silicon) or "pytorch"
    backend: str = "mlx"

    # Model preset: "qwen3-coder", "gpt-oss", or None for custom
    preset: str | None = "qwen3-coder"

    # Custom model settings (used if preset is None)
    base_model: str = "Qwen/Qwen3-Coder-30B-A3B"
    adapter_path: str | None = None

    # Runtime settings
    timeout_seconds: float = 30.0  # Increased for initial load
    max_memory_gb: float = 64.0  # MLX models use more memory

    # Resource directories
    patterns_dir: str | None = None
    conventions_dir: str | None = None


class MochiMCPServer:
    """MCP Server for Mochi inference.

    Implements JSON-RPC 2.0 over stdio (L-mcp-compliance).
    """

    # Tool definitions
    TOOLS: list[ToolDefinition] = [
        ToolDefinition(
            name="domain_query",
            description=(
                "Query the domain-specific model for code generation. "
                "Supports both code completion and document-to-code generation. "
                "Use code-style instructions: 'Implement the following', 'Write the implementation'. "
                "WARNING: Output may contain incorrect schema/API names - verify against actual source. "
                "MODES: 'auto' (default) retries with conservative params if confidence < 0.5, "
                "'conservative' for precise short output, 'creative' for full implementations. "
                "FORMAT: ChatML (default) for general use, Alpaca (use_chatml=false) for short completions."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": (
                            "Code completion instruction. Use: 'Fill in the typescript code', "
                            "'Implement the following typescript code based on the context', "
                            "'Write the implementation for this typescript function'"
                        ),
                    },
                    "input": {
                        "type": "string",
                        "description": (
                            "Code context with file path comment. "
                            "Format: '// File: path/to/file.ts\\n<code before completion point>'"
                        ),
                    },
                    "context": {
                        "type": "string",
                        "description": (
                            "LSP context with available methods/types. "
                            "Format: '// Methods on TypeName:\\n//   method1()\\n//   method2()'. "
                            "Used to validate output and prevent API hallucinations."
                        ),
                    },
                    "max_tokens": {
                        "type": "number",
                        "description": "Maximum tokens to generate (default: 2048)",
                    },
                    "min_tokens": {
                        "type": "number",
                        "description": "Minimum tokens to generate - prevents too-short outputs (default: 10)",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature 0.0-1.0 (default: 0.1, deterministic)",
                    },
                    "validate": {
                        "type": "boolean",
                        "description": "Validate output against context for hallucinations (default: true if context provided)",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["auto", "conservative", "creative"],
                        "description": "Generation mode: 'auto' (default), 'conservative' (1-line completions), 'creative' (full functions, error handling)",
                    },
                    "auto_retry": {
                        "type": "boolean",
                        "description": "Auto-retry with conservative mode if confidence is low (default: true in auto mode)",
                    },
                    "use_chatml": {
                        "type": "boolean",
                        "description": "Use ChatML format (default: true, Qwen3 native). Set false for Alpaca format (short completions only)",
                    },
                },
                "required": ["instruction"],
            },
        ),
        ToolDefinition(
            name="complete_code",
            description=(
                "Complete TypeScript/JavaScript code using kiri domain patterns. "
                "Provide code prefix (and optional suffix for fill-in-the-middle). "
                "Model generates continuation matching kiri coding conventions. "
                "NOTE: Generated code follows kiri STYLE but may have incorrect "
                "API/schema names. Verify against actual source before use."
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
                    "language": {
                        "type": "string",
                        "description": "Programming language (default: typescript)",
                    },
                },
                "required": ["prefix"],
            },
        ),
    ]

    def __init__(self, config: ServerConfig | None = None) -> None:
        """Initialize MCP server.

        Args:
            config: Server configuration
        """
        self.config = config or ServerConfig()
        self.engine: InferenceEngineProtocol | None = None
        self._initialized = False
        self._backend = self.config.backend

        # Resource storage
        self._patterns: dict[str, str] = {}
        self._conventions: dict[str, str] = {}

    def initialize(self) -> None:
        """Initialize the server and load model.

        Raises:
            RuntimeError: If initialization fails
        """
        if self._initialized:
            return

        print("Loading model...", file=sys.stderr)

        if self.config.backend == "mlx":
            self._initialize_mlx()
        else:
            self._initialize_pytorch()

        # Load patterns if directory specified
        if self.config.patterns_dir:
            self._load_patterns(Path(self.config.patterns_dir))

        # Load conventions if directory specified
        if self.config.conventions_dir:
            self._load_conventions(Path(self.config.conventions_dir))

        self._initialized = True
        print("Model loaded.", file=sys.stderr)

    def _ensure_initialized(self) -> None:
        """Ensure model is loaded (lazy initialization)."""
        if not self._initialized:
            self.initialize()

    def _initialize_mlx(self) -> None:
        """Initialize MLX backend (recommended for Apple Silicon)."""
        from mochi.mcp.inference_mlx import MLXInferenceEngine

        self.engine = MLXInferenceEngine(
            preset=self.config.preset,
            model_path=self.config.base_model if not self.config.preset else None,
            adapter_path=self.config.adapter_path,
            timeout_seconds=self.config.timeout_seconds,
            max_memory_gb=self.config.max_memory_gb,
        )
        self.engine.load()

    def _initialize_pytorch(self) -> None:
        """Initialize PyTorch backend (fallback)."""
        from mochi.mcp.inference import InferenceEngine

        self.engine = InferenceEngine(
            base_model=self.config.base_model,
            adapter_path=self.config.adapter_path,
            timeout_seconds=self.config.timeout_seconds,
            max_memory_gb=self.config.max_memory_gb,
        )
        if self.config.adapter_path:
            self.engine.load()

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
                "version": "0.1.0",
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

        if tool_name == "domain_query":
            return self._handle_domain_query(arguments)
        elif tool_name == "complete_code":
            return self._handle_complete_code(arguments)
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

        if not self.engine or not self.engine.is_loaded:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Model not loaded. Server not properly initialized.",
                    }
                ],
                "isError": True,
            }

        instruction = args.get("instruction", "")
        input_text = args.get("input", "")
        context = args.get("context", "")
        max_tokens = args.get("max_tokens", 2048)
        min_tokens = args.get("min_tokens", 10)  # P0: 最小出力長
        temperature = args.get("temperature", 0.1)
        validate = args.get("validate", bool(context))  # Default to True if context provided
        mode_str = args.get("mode", "auto")  # P2: モード切替
        auto_retry = args.get("auto_retry", True)  # P2: 自動リトライ
        use_chatml = args.get("use_chatml", True)  # ChatML is default (Qwen3 native)

        # P2: Import and convert mode string to enum
        from mochi.mcp.inference_mlx import GenerationMode
        mode_map = {
            "auto": GenerationMode.AUTO,
            "conservative": GenerationMode.CONSERVATIVE,
            "creative": GenerationMode.CREATIVE,
        }
        mode = mode_map.get(mode_str, GenerationMode.AUTO)

        try:
            # Pass context to engine for context-aware generation
            result = self.engine.generate(
                instruction=instruction,
                input_text=input_text,
                context=context,
                max_new_tokens=max_tokens,
                min_new_tokens=min_tokens,  # P0: Use min_tokens
                temperature=temperature,
                mode=mode,  # P2: Pass mode
                auto_retry=auto_retry,  # P2: Pass auto_retry
                use_chatml=use_chatml,  # ChatML for longer generation
            )

            response_data = {
                "response": result.response,
                "confidence": result.confidence,
                "inference_time_ms": result.inference_time_ms,
                "tokens_generated": result.tokens_generated,
                "mode_used": result.mode_used,  # P2: 使用されたモード
                "retried": result.retried,  # P2: リトライされたか
            }

            # P0: Add confidence warnings
            if result.warning:
                response_data["warning"] = result.warning
            if result.alternative_action:
                response_data["alternative_action"] = result.alternative_action

            # Validate output against context if requested
            if validate and context:
                validator = OutputValidator()
                validation = validator.validate(result.response, context)
                response_data["validation"] = {
                    "is_valid": validation.is_valid,
                    "hallucination_rate": validation.hallucination_rate,
                }
                if validation.hallucinated_methods:
                    response_data["validation"]["hallucinated_methods"] = validation.hallucinated_methods
                    # Add suggestions for corrections
                    suggestions = validator.suggest_corrections(
                        validation.hallucinated_methods,
                        validation.available_methods,
                    )
                    if any(suggestions.values()):
                        response_data["validation"]["suggestions"] = suggestions

            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(response_data, indent=2, ensure_ascii=False),
                    }
                ]
            }

        except TimeoutError as e:
            return {
                "content": [{"type": "text", "text": f"Timeout: {e}"}],
                "isError": True,
            }
        except MemoryError as e:
            return {
                "content": [{"type": "text", "text": f"Memory error: {e}"}],
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

        if not self.engine or not self.engine.is_loaded:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Model not loaded. Server not properly initialized.",
                    }
                ],
                "isError": True,
            }

        prefix = args.get("prefix", "")
        suffix = args.get("suffix", "")

        try:
            completions = self.engine.complete(
                prefix=prefix,
                suffix=suffix,
                max_new_tokens=256,
                num_alternatives=3,
            )

            response_data = {
                "completion": completions[0] if completions else "",
                "alternatives": completions[1:] if len(completions) > 1 else [],
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
            "name": "Model Statistics",
            "mimeType": "application/json",
            "description": "Training statistics and model info",
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
                "preset": self.config.preset,
                "base_model": self.config.base_model,
                "adapter_path": str(self.config.adapter_path) if self.config.adapter_path else None,
                "model_loaded": self.engine.is_loaded if self.engine else False,
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
        """Run server using stdio transport (L-mcp-compliance).

        Note: Model loading is deferred to first tool call for faster MCP initialization.
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
        if self.engine:
            self.engine.unload()
        self._initialized = False


if __name__ == "__main__":
    server = MochiMCPServer()
    try:
        server.run_stdio()
    finally:
        server.shutdown()
