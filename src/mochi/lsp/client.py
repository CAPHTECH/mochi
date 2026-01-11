"""LSP Client for context extraction.

Communicates with language servers (tsserver, pylsp, etc.) via JSON-RPC
to extract completion candidates, type information, and workspace symbols.

Law compliance:
- L-fallback-graceful: All LSP operations have timeouts and exception handling
- L-batch-efficiency: Connection reuse and caching for batch processing
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CompletionItemKind(IntEnum):
    """LSP CompletionItemKind values."""

    Text = 1
    Method = 2
    Function = 3
    Constructor = 4
    Field = 5
    Variable = 6
    Class = 7
    Interface = 8
    Module = 9
    Property = 10
    Unit = 11
    Value = 12
    Enum = 13
    Keyword = 14
    Snippet = 15
    Color = 16
    File = 17
    Reference = 18
    Folder = 19
    EnumMember = 20
    Constant = 21
    Struct = 22
    Event = 23
    Operator = 24
    TypeParameter = 25


class SymbolKind(IntEnum):
    """LSP SymbolKind values."""

    File = 1
    Module = 2
    Namespace = 3
    Package = 4
    Class = 5
    Method = 6
    Property = 7
    Field = 8
    Constructor = 9
    Enum = 10
    Interface = 11
    Function = 12
    Variable = 13
    Constant = 14
    String = 15
    Number = 16
    Boolean = 17
    Array = 18
    Object = 19
    Key = 20
    Null = 21
    EnumMember = 22
    Struct = 23
    Event = 24
    Operator = 25
    TypeParameter = 26


@dataclass
class CompletionItem:
    """Completion item from LSP."""

    label: str
    kind: CompletionItemKind
    detail: str | None = None
    documentation: str | None = None
    insert_text: str | None = None


@dataclass
class HoverInfo:
    """Hover information from LSP."""

    contents: str
    range_start: tuple[int, int] | None = None
    range_end: tuple[int, int] | None = None


@dataclass
class SymbolInfo:
    """Workspace symbol from LSP."""

    name: str
    kind: SymbolKind
    location_uri: str
    location_range: tuple[tuple[int, int], tuple[int, int]] | None = None
    container_name: str | None = None


@dataclass
class LSPClient:
    """Language Server Protocol client for context extraction.

    Manages communication with language servers via JSON-RPC over stdio.
    Supports TypeScript (tsserver), Python (pylsp), and other LSP-compliant servers.

    Usage:
        async with LSPClient("typescript", project_root) as client:
            completions = await client.get_completions(file, line, col)

    Terms:
    - JSON-RPC: Wire protocol for LSP communication
    - tsserver: TypeScript language server
    - pylsp: Python language server
    """

    language: str
    project_root: Path
    timeout_seconds: float = 5.0

    _process: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _request_id: int = field(default=0, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)
    _open_files: set[str] = field(default_factory=set, init=False, repr=False)

    # Language server commands
    _SERVER_COMMANDS: dict[str, list[str]] = field(
        default_factory=lambda: {
            "typescript": ["npx", "typescript-language-server", "--stdio"],
            "javascript": ["npx", "typescript-language-server", "--stdio"],
            "python": ["pylsp"],
            "rust": ["rust-analyzer"],
            "go": ["gopls", "serve"],
        },
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root)
        self._open_files = set()

    async def __aenter__(self) -> LSPClient:
        """Start LSP server and initialize connection."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Shutdown LSP server."""
        await self.stop()

    def _get_server_command(self) -> list[str]:
        """Get the LSP server command for the language."""
        if self.language not in self._SERVER_COMMANDS:
            raise ValueError(
                f"Unsupported language: {self.language}. "
                f"Supported: {list(self._SERVER_COMMANDS.keys())}"
            )
        return self._SERVER_COMMANDS[self.language]

    async def start(self) -> None:
        """Start the LSP server process."""
        if self._process is not None:
            return

        cmd = self._get_server_command()
        logger.info(f"Starting LSP server: {' '.join(cmd)}")

        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.project_root),
            )
        except FileNotFoundError as e:
            logger.error(f"LSP server command not found: {cmd[0]}")
            raise RuntimeError(
                f"LSP server not available: {cmd[0]}. "
                f"Install with: npm install -g typescript-language-server typescript"
            ) from e

        # Initialize the LSP connection
        await self._initialize()

    async def stop(self) -> None:
        """Stop the LSP server process."""
        if self._process is None:
            return

        try:
            # Send shutdown request
            await self._send_request("shutdown", {})
            # Send exit notification
            self._send_notification("exit", {})
        except Exception as e:
            logger.warning(f"Error during LSP shutdown: {e}")
        finally:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            self._initialized = False
            self._open_files.clear()

    async def _initialize(self) -> None:
        """Send LSP initialize request."""
        params = {
            "processId": None,
            "rootUri": f"file://{self.project_root}",
            "rootPath": str(self.project_root),
            "capabilities": {
                "textDocument": {
                    "completion": {
                        "completionItem": {
                            "snippetSupport": False,
                            "documentationFormat": ["plaintext", "markdown"],
                        }
                    },
                    "hover": {
                        "contentFormat": ["plaintext", "markdown"],
                    },
                    "synchronization": {
                        "didOpen": True,
                        "didClose": True,
                    },
                },
                "workspace": {
                    "symbol": {
                        "symbolKind": {
                            "valueSet": list(range(1, 27)),
                        }
                    }
                },
            },
        }

        result = await self._send_request("initialize", params)
        logger.debug(f"LSP initialized: {result.get('serverInfo', {})}")

        # Send initialized notification
        self._send_notification("initialized", {})
        self._initialized = True

    def _send_notification(self, method: str, params: dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("LSP server not running")

        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        content = json.dumps(message)
        header = f"Content-Length: {len(content)}\r\n\r\n"
        full_message = header + content

        self._process.stdin.write(full_message.encode())
        self._process.stdin.flush()

    async def _send_request(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Send a JSON-RPC request and wait for response."""
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("LSP server not running")

        self._request_id += 1
        request_id = self._request_id

        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        content = json.dumps(message)
        header = f"Content-Length: {len(content)}\r\n\r\n"
        full_message = header + content

        self._process.stdin.write(full_message.encode())
        self._process.stdin.flush()

        # Read response with timeout
        return await asyncio.wait_for(
            self._read_response(request_id),
            timeout=self.timeout_seconds,
        )

    async def _read_response(self, expected_id: int) -> dict[str, Any]:
        """Read JSON-RPC response from server."""
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("LSP server not running")

        loop = asyncio.get_event_loop()

        while True:
            # Read headers
            headers = {}
            while True:
                line = await loop.run_in_executor(
                    None, self._process.stdout.readline
                )
                line = line.decode().strip()
                if not line:
                    break
                if ":" in line:
                    key, value = line.split(":", 1)
                    headers[key.strip()] = value.strip()

            if "Content-Length" not in headers:
                continue

            content_length = int(headers["Content-Length"])
            content = await loop.run_in_executor(
                None, self._process.stdout.read, content_length
            )

            try:
                response = json.loads(content.decode())
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON response: {e}")
                continue

            # Check if this is the response we're waiting for
            if response.get("id") == expected_id:
                if "error" in response:
                    error = response["error"]
                    raise RuntimeError(
                        f"LSP error {error.get('code')}: {error.get('message')}"
                    )
                return response.get("result", {})

            # Log notifications and other messages
            if "method" in response:
                logger.debug(f"LSP notification: {response['method']}")

    async def open_file(self, file_path: Path) -> None:
        """Open a file in the LSP server."""
        uri = f"file://{file_path.absolute()}"
        if uri in self._open_files:
            return

        content = file_path.read_text()

        # Determine language ID
        suffix = file_path.suffix.lower()
        language_id_map = {
            ".ts": "typescript",
            ".tsx": "typescriptreact",
            ".js": "javascript",
            ".jsx": "javascriptreact",
            ".py": "python",
            ".rs": "rust",
            ".go": "go",
        }
        language_id = language_id_map.get(suffix, self.language)

        self._send_notification(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": language_id,
                    "version": 1,
                    "text": content,
                }
            },
        )
        self._open_files.add(uri)
        # Give the server time to process
        await asyncio.sleep(0.1)

    async def close_file(self, file_path: Path) -> None:
        """Close a file in the LSP server."""
        uri = f"file://{file_path.absolute()}"
        if uri not in self._open_files:
            return

        self._send_notification(
            "textDocument/didClose",
            {"textDocument": {"uri": uri}},
        )
        self._open_files.discard(uri)

    async def get_completions(
        self, file_path: Path, line: int, character: int
    ) -> list[CompletionItem]:
        """Get completion items at a specific position.

        Args:
            file_path: Path to the source file
            line: 0-indexed line number
            character: 0-indexed character position

        Returns:
            List of completion items available at the position
        """
        await self.open_file(file_path)

        uri = f"file://{file_path.absolute()}"
        params = {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
        }

        try:
            result = await self._send_request("textDocument/completion", params)
        except (asyncio.TimeoutError, RuntimeError) as e:
            logger.warning(f"Completion request failed: {e}")
            return []

        # Handle both list and CompletionList response
        items = result if isinstance(result, list) else result.get("items", [])

        completions = []
        for item in items:
            completions.append(
                CompletionItem(
                    label=item.get("label", ""),
                    kind=CompletionItemKind(item.get("kind", 1)),
                    detail=item.get("detail"),
                    documentation=self._extract_documentation(
                        item.get("documentation")
                    ),
                    insert_text=item.get("insertText"),
                )
            )

        return completions

    async def get_hover(
        self, file_path: Path, line: int, character: int
    ) -> HoverInfo | None:
        """Get hover information at a specific position.

        Args:
            file_path: Path to the source file
            line: 0-indexed line number
            character: 0-indexed character position

        Returns:
            HoverInfo if available, None otherwise
        """
        await self.open_file(file_path)

        uri = f"file://{file_path.absolute()}"
        params = {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
        }

        try:
            result = await self._send_request("textDocument/hover", params)
        except (asyncio.TimeoutError, RuntimeError) as e:
            logger.warning(f"Hover request failed: {e}")
            return None

        if not result or "contents" not in result:
            return None

        contents = self._extract_hover_contents(result["contents"])
        range_info = result.get("range")

        return HoverInfo(
            contents=contents,
            range_start=(
                (range_info["start"]["line"], range_info["start"]["character"])
                if range_info
                else None
            ),
            range_end=(
                (range_info["end"]["line"], range_info["end"]["character"])
                if range_info
                else None
            ),
        )

    async def get_workspace_symbols(self, query: str = "") -> list[SymbolInfo]:
        """Get workspace symbols matching a query.

        Args:
            query: Symbol name filter (empty for all symbols)

        Returns:
            List of symbols in the workspace
        """
        params = {"query": query}

        try:
            result = await self._send_request("workspace/symbol", params)
        except (asyncio.TimeoutError, RuntimeError) as e:
            logger.warning(f"Workspace symbol request failed: {e}")
            return []

        symbols = []
        for item in result or []:
            location = item.get("location", {})
            range_info = location.get("range")

            symbols.append(
                SymbolInfo(
                    name=item.get("name", ""),
                    kind=SymbolKind(item.get("kind", 1)),
                    location_uri=location.get("uri", ""),
                    location_range=(
                        (
                            (
                                range_info["start"]["line"],
                                range_info["start"]["character"],
                            ),
                            (
                                range_info["end"]["line"],
                                range_info["end"]["character"],
                            ),
                        )
                        if range_info
                        else None
                    ),
                    container_name=item.get("containerName"),
                )
            )

        return symbols

    def _extract_documentation(self, doc: Any) -> str | None:
        """Extract documentation string from various formats."""
        if doc is None:
            return None
        if isinstance(doc, str):
            return doc
        if isinstance(doc, dict):
            return doc.get("value", str(doc))
        return str(doc)

    def _extract_hover_contents(self, contents: Any) -> str:
        """Extract hover contents from various formats."""
        if isinstance(contents, str):
            return contents
        if isinstance(contents, dict):
            return contents.get("value", str(contents))
        if isinstance(contents, list):
            parts = []
            for item in contents:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(item.get("value", str(item)))
            return "\n".join(parts)
        return str(contents)
