"""Context extractor using LSP for training data generation.

Extracts relevant context (available methods, types, schema info) from
source code positions and formats them as Context Blocks for training examples.

Law compliance:
- L-fallback-graceful: Returns empty context on LSP failures
- L-context-format: Uses "// Available methods: ..." format
- L-batch-efficiency: Caching and batch processing support
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from mochi.lsp.client import (
    CompletionItem,
    CompletionItemKind,
    LSPClient,
    SymbolInfo,
    SymbolKind,
)

logger = logging.getLogger(__name__)

# Global builtin functions to filter out from LSP completions
# These pollute context and cause hallucinations when model sees them
GLOBAL_BUILTINS_FILTER: frozenset[str] = frozenset({
    # JavaScript global functions
    "atob", "btoa", "eval", "fetch", "setTimeout", "setInterval",
    "clearTimeout", "clearInterval", "clearImmediate", "setImmediate",
    "alert", "confirm", "prompt", "print",
    "parseInt", "parseFloat", "isNaN", "isFinite",
    "encodeURI", "decodeURI", "encodeURIComponent", "decodeURIComponent",
    "escape", "unescape", "postMessage",
    # Web Platform APIs (pollute TypeScript completions)
    "queueMicrotask", "structuredClone", "reportError",
    "requestAnimationFrame", "cancelAnimationFrame",
    "requestIdleCallback", "cancelIdleCallback",
    "addEventListener", "removeEventListener", "dispatchEvent",
    "getComputedStyle", "matchMedia", "scroll", "scrollTo", "scrollBy",
    "open", "close", "focus", "blur", "moveBy", "moveTo", "resizeBy", "resizeTo",
    # JavaScript global constructors
    "Object", "Array", "String", "Number", "Boolean", "Date", "Function",
    "Math", "JSON", "Promise", "RegExp", "Error", "Symbol", "BigInt",
    "Map", "Set", "WeakMap", "WeakSet",
    "ArrayBuffer", "DataView", "Int8Array", "Uint8Array", "Int16Array",
    "Uint16Array", "Int32Array", "Uint32Array", "Float32Array", "Float64Array",
    "Proxy", "Reflect", "SharedArrayBuffer", "Atomics",
    "FinalizationRegistry", "WeakRef",
    # Node.js globals
    "require", "module", "exports", "__dirname", "__filename",
    "process", "Buffer", "global", "globalThis",
    # Console methods (if appearing as top-level)
    "console", "log", "warn", "error", "info", "debug",
    # Common DOM globals
    "document", "window", "navigator", "location", "history",
    "localStorage", "sessionStorage", "XMLHttpRequest",
    "Request", "Response", "Headers", "URL", "URLSearchParams",
    "FormData", "Blob", "File", "FileReader", "FileList",
    "Image", "Audio", "Video", "MediaSource",
    "Worker", "ServiceWorker", "SharedWorker",
    "WebSocket", "EventSource", "BroadcastChannel",
    "Notification", "PushManager",
    "Cache", "CacheStorage", "IndexedDB", "IDBDatabase",
    "AbortController", "AbortSignal",
    "TextEncoder", "TextDecoder",
    "Crypto", "SubtleCrypto", "CryptoKey",
    "Performance", "PerformanceObserver",
    "IntersectionObserver", "MutationObserver", "ResizeObserver",
    "CustomEvent", "Event", "MessageEvent", "ErrorEvent",
    "DOMParser", "XMLSerializer",
    # TypeScript utility types (often pollute completions)
    "Partial", "Required", "Readonly", "Record", "Pick", "Omit",
    "Exclude", "Extract", "NonNullable", "ReturnType", "Parameters",
    "ConstructorParameters", "InstanceType", "ThisType",
    "Awaited", "Uppercase", "Lowercase", "Capitalize", "Uncapitalize",
    # Vitest/Jest globals
    "describe", "it", "test", "expect", "beforeEach", "afterEach",
    "beforeAll", "afterAll", "vi", "jest", "mock", "spyOn",
    # Node.js fs/stream methods (appear in completions)
    "access", "accessSync", "appendFile", "appendFileSync",
    "chmod", "chmodSync", "chown", "chownSync", "close", "closeSync",
    "copyFile", "copyFileSync", "cp", "cpSync",
    "createReadStream", "createWriteStream",
    "exists", "existsSync", "fchmod", "fchmodSync", "fchown", "fchownSync",
    "fdatasync", "fdatasyncSync", "fstat", "fstatSync",
    "fsync", "fsyncSync", "ftruncate", "ftruncateSync",
    "futimes", "futimesSync", "lchmod", "lchmodSync", "lchown", "lchownSync",
    "link", "linkSync", "lstat", "lstatSync",
    "mkdir", "mkdirSync", "mkdtemp", "mkdtempSync",
    "open", "openSync", "opendir", "opendirSync",
    "read", "readSync", "readdir", "readdirSync",
    "readFile", "readFileSync", "readlink", "readlinkSync", "readv", "readvSync",
    "realpath", "realpathSync", "rename", "renameSync",
    "rm", "rmSync", "rmdir", "rmdirSync",
    "stat", "statSync", "statfs", "statfsSync",
    "symlink", "symlinkSync", "truncate", "truncateSync",
    "unlink", "unlinkSync", "unwatchFile", "utimes", "utimesSync",
    "watch", "watchFile", "write", "writeSync",
    "writeFile", "writeFileSync", "writev", "writevSync",
    # Node.js AbortController/Signal methods
    "abort", "aborted", "throwIfAborted",
    "addAbortListener", "addAbortSignal",
    # Node.js events methods
    "addListener", "emit", "eventNames", "getMaxListeners",
    "listenerCount", "listeners", "off", "on", "once",
    "prependListener", "prependOnceListener", "rawListeners",
    "removeAllListeners", "removeListener", "setMaxListeners",
    # Node.js stream methods
    "pipe", "unpipe", "cork", "uncork", "destroy",
    "pause", "resume", "read", "push", "unshift",
    "wrap", "compose", "pipeline", "finished",
    # Node.js crypto methods
    "createHash", "createHmac", "createCipheriv", "createDecipheriv",
    "createSign", "createVerify", "generateKey", "generateKeyPair",
    "randomBytes", "randomFill", "randomInt", "randomUUID",
    "scrypt", "scryptSync", "pbkdf2", "pbkdf2Sync",
    # Node.js path methods
    "basename", "dirname", "extname", "format", "isAbsolute",
    "join", "normalize", "parse", "relative", "resolve",
    "sep", "delimiter", "posix", "win32",
    # TypeScript compiler API methods (pollute completions)
    "addEmitHelper", "addEmitHelpers", "addRange",
    "addSyntheticLeadingComment", "addSyntheticTrailingComment",
    "after", "before", "between",
    "acquireLock", "releaseLock",
})

# Global/standard library types to filter out
# These are common across all projects and not domain-specific
GLOBAL_TYPES_FILTER: frozenset[str] = frozenset({
    # JavaScript built-in types
    "Object", "Array", "String", "Number", "Boolean", "Function", "Symbol",
    "Error", "TypeError", "RangeError", "SyntaxError", "ReferenceError",
    "EvalError", "URIError", "AggregateError",
    "Promise", "PromiseLike", "Thenable",
    "Map", "Set", "WeakMap", "WeakSet", "ReadonlyMap", "ReadonlySet",
    "ArrayBuffer", "SharedArrayBuffer", "DataView",
    "Int8Array", "Uint8Array", "Uint8ClampedArray",
    "Int16Array", "Uint16Array", "Int32Array", "Uint32Array",
    "Float32Array", "Float64Array", "BigInt64Array", "BigUint64Array",
    "RegExp", "RegExpMatchArray", "RegExpExecArray",
    "Date", "JSON",
    "Proxy", "ProxyHandler", "Reflect",
    # Node.js built-in types
    "Buffer", "NodeJS", "Process", "Global",
    "EventEmitter", "Readable", "Writable", "Duplex", "Transform",
    "IncomingMessage", "ServerResponse", "ClientRequest",
    "Server", "Socket", "Agent", "TLSSocket",
    "ChildProcess", "Cluster", "Worker",
    "Console", "Inspector",
    "ReadStream", "WriteStream", "Stats", "Dirent",
    "URL", "URLSearchParams",
    "AsyncLocalStorage", "AsyncResource",
    "AssertionError",
    # TypeScript utility types
    "Partial", "Required", "Readonly", "Record", "Pick", "Omit",
    "Exclude", "Extract", "NonNullable", "ReturnType", "Parameters",
    "ConstructorParameters", "InstanceType", "ThisType", "ThisParameterType",
    "OmitThisParameter", "Awaited", "NoInfer",
    "Uppercase", "Lowercase", "Capitalize", "Uncapitalize",
    "PropertyKey", "PropertyDescriptor", "PropertyDescriptorMap",
    # Common library types (too generic)
    "Iterator", "IterableIterator", "Generator", "GeneratorFunction",
    "AsyncIterator", "AsyncIterableIterator", "AsyncGenerator",
    # DOM types
    "Document", "Element", "Node", "NodeList", "HTMLElement",
    "Event", "EventTarget", "EventListener",
    "Window", "Navigator", "Location", "History",
    "Request", "Response", "Headers", "Body",
    "Blob", "File", "FileList", "FileReader",
    "FormData", "URLSearchParams",
    "AbortController", "AbortSignal",
    "MessagePort", "MessageChannel",
    "Performance", "PerformanceEntry",
    # Test framework types
    "Mock", "SpyInstance", "Mocked",
})

# Patterns to identify library-internal symbols
# Symbols matching these patterns are usually not project-specific
LIBRARY_INTERNAL_PATTERNS: tuple[str, ...] = (
    r"^\$",           # Zod internal: $ZodAsyncError, etc.
    r"^_[a-z]",       # Private/internal: _any, _array, etc.
    r"^__",           # Dunder methods: __proto__, etc.
    r"Internal$",     # *Internal types
    r"Private$",      # *Private types
    r"^NodeJS\.",     # NodeJS namespace
    r"^globalThis\.", # globalThis namespace
)


@dataclass
class MethodSignature:
    """Structured method signature information.

    Stores full method signature details extracted from LSP.
    """

    name: str
    parameters: str = ""
    return_type: str = ""
    documentation: str = ""

    def format_short(self) -> str:
        """Format as short signature: name(params): ReturnType"""
        sig = self.name
        if self.parameters:
            sig += f"({self.parameters})"
        if self.return_type:
            sig += f": {self.return_type}"
        return sig

    def format_full(self) -> str:
        """Format with documentation if available."""
        sig = self.format_short()
        if self.documentation:
            # Take first line of documentation
            doc_line = self.documentation.split('\n')[0].strip()
            if doc_line:
                sig += f" - {doc_line[:80]}"
        return sig


@dataclass
class TypeInfo:
    """Type information extracted from LSP.

    Stores type/interface/class information.
    """

    name: str
    kind: str = "type"  # type, interface, class, enum
    members: list[str] = field(default_factory=list)

    def format(self) -> str:
        """Format type info."""
        if self.members:
            return f"{self.name} {{ {', '.join(self.members[:5])} }}"
        return self.name


@dataclass
class SchemaInfo:
    """Database schema information."""

    tables: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> SchemaInfo:
        """Load schema from YAML file."""
        if not path.exists():
            return cls()
        with path.open() as f:
            data = yaml.safe_load(f) or {}
        return cls(tables=data.get("tables", []))

    def format(self) -> str:
        """Format schema as context string."""
        if not self.tables:
            return ""
        parts = []
        for table in self.tables:
            name = table.get("name", "")
            columns = table.get("columns", [])
            parts.append(f"{name}({', '.join(columns)})")
        return ", ".join(parts)


@dataclass
class ContextBlock:
    """Formatted context block for training data.

    Enhanced to include full method signatures and type information.
    """

    methods: list[MethodSignature] = field(default_factory=list)
    types: list[TypeInfo] = field(default_factory=list)
    schema: str = ""
    imports: list[str] = field(default_factory=list)
    # Legacy support: simple string lists
    method_names: list[str] = field(default_factory=list)
    type_names: list[str] = field(default_factory=list)
    # Receiver type info (e.g., "DuckDBClient", "UserRepository")
    receiver_type: str | None = None

    def format(self, detailed: bool = True) -> str:
        """Format as context block string.

        Args:
            detailed: If True, include full signatures. If False, use simple names.

        Returns context in the format (detailed=True):
        ```
        // Methods on DuckDBClient:
        //   all<T>(sql: string): Promise<T[]>
        //   run(sql: string): Promise<void>
        // Available types: DuckDBClient, QueryResult
        // DB schema: table1(col1, col2), table2(...)
        ```

        Or (detailed=False):
        ```
        // Available methods: all, run, prepare
        // Available types: DuckDBClient, QueryResult
        ```
        """
        lines = []

        if detailed and self.methods:
            # Use receiver type if available for clearer context
            if self.receiver_type:
                lines.append(f"// Methods on {self.receiver_type}:")
            else:
                lines.append("// Available methods:")
            for method in self.methods[:15]:
                lines.append(f"//   {method.format_short()}")
        elif self.methods or self.method_names:
            # Fallback to simple names
            names = self.method_names or [m.name for m in self.methods]
            methods_str = ", ".join(names[:15])
            lines.append(f"// Available methods: {methods_str}")

        if self.types or self.type_names:
            if detailed and self.types:
                type_strs = [t.format() for t in self.types[:10]]
            else:
                type_strs = self.type_names or [t.name for t in self.types]
            types_str = ", ".join(type_strs[:15])
            lines.append(f"// Available types: {types_str}")

        if self.schema:
            lines.append(f"// DB schema: {self.schema}")

        if self.imports:
            imports_str = ", ".join(self.imports[:10])
            lines.append(f"// Imports: {imports_str}")

        return "\n".join(lines)

    def format_compact(self) -> str:
        """Format as compact single-line context (for backward compatibility)."""
        return self.format(detailed=False)

    def is_empty(self) -> bool:
        """Check if context block has any content."""
        return not (
            self.methods or self.types or self.schema or self.imports
            or self.method_names or self.type_names
        )


class ContextExtractor:
    """Extract context information using LSP for training data.

    Terms:
    - ContextBlock: Formatted context for training examples
    - SchemaInfo: Database schema information

    Laws:
    - L-fallback-graceful: LSP failures return empty context
    - L-context-format: Uses standard comment format
    """

    def __init__(
        self,
        lsp_client: LSPClient,
        schema_path: Path | None = None,
        cache_size: int = 1000,
    ) -> None:
        """Initialize context extractor.

        Args:
            lsp_client: LSP client for language server communication
            schema_path: Optional path to schema.yaml file
            cache_size: Size of the symbol cache
        """
        self.lsp = lsp_client
        self.schema = (
            SchemaInfo.from_yaml(schema_path) if schema_path else SchemaInfo()
        )
        self._cache_size = cache_size
        self._workspace_symbols_cache: list[SymbolInfo] | None = None

    async def extract_at_position(
        self,
        file_path: Path,
        line: int,
        character: int,
        include_schema: bool = True,
    ) -> ContextBlock:
        """Extract context at a specific code position.

        Args:
            file_path: Path to the source file
            line: 0-indexed line number
            character: 0-indexed character position
            include_schema: Whether to include schema information

        Returns:
            ContextBlock with available methods, types, etc.
        """
        context = ContextBlock()

        try:
            # Try to detect receiver type (for "obj." positions)
            receiver_type = await self._detect_receiver_type(file_path, line, character)
            if receiver_type:
                context.receiver_type = receiver_type

            # Get completions at position
            completions = await self.lsp.get_completions(file_path, line, character)
            context.methods = self._extract_methods(completions)
            context.types = self._extract_types(completions)

            # Get workspace symbols for imports
            if self._workspace_symbols_cache is None:
                self._workspace_symbols_cache = await self.lsp.get_workspace_symbols()
            context.imports = self._extract_imports(self._workspace_symbols_cache)

            # Add schema if requested
            if include_schema and self.schema.tables:
                context.schema = self.schema.format()

        except Exception as e:
            # L-fallback-graceful: Log and return partial/empty context
            logger.warning(f"Context extraction failed at {file_path}:{line}: {e}")

        return context

    async def _detect_receiver_type(
        self,
        file_path: Path,
        line: int,
        character: int,
    ) -> str | None:
        """Detect the type of the receiver object before a dot.

        For position at "db.|", detects the type of "db" using LSP hover.

        Args:
            file_path: Path to the source file
            line: 0-indexed line number
            character: 0-indexed character position (should be after a dot)

        Returns:
            Type name if detected (e.g., "DuckDBClient"), None otherwise
        """
        try:
            content = file_path.read_text()
            lines = content.split("\n")

            if line >= len(lines):
                return None

            line_text = lines[line]

            # Check if position is after a dot
            if character == 0 or character > len(line_text):
                return None

            # Find the dot position (should be just before character)
            dot_pos = character - 1
            if dot_pos >= 0 and dot_pos < len(line_text) and line_text[dot_pos] == ".":
                # Find the identifier before the dot
                start = dot_pos - 1
                while start >= 0 and (line_text[start].isalnum() or line_text[start] in "_$"):
                    start -= 1
                start += 1

                if start < dot_pos:
                    # Variable name before the dot
                    var_name = line_text[start:dot_pos]

                    # Use LSP hover to get type
                    hover = await self.lsp.get_hover(file_path, line, start)
                    if hover and hover.contents:
                        return self._parse_type_from_hover(hover.contents, var_name)

        except Exception as e:
            logger.debug(f"Receiver type detection failed: {e}")

        return None

    def _parse_type_from_hover(self, hover_contents: str, var_name: str) -> str | None:
        """Parse type name from hover contents.

        TypeScript hover typically returns:
        - "(const) db: DuckDBClient"
        - "let client: HttpClient"
        - "(parameter) repo: UserRepository"

        Args:
            hover_contents: Raw hover contents from LSP
            var_name: Variable name (for context)

        Returns:
            Extracted type name or None
        """
        if not hover_contents:
            return None

        # Try to match "varname: TypeName" pattern
        # Handle patterns like:
        # - "const db: DuckDBClient"
        # - "(const) db: DuckDBClient"
        # - "let users: User[]"
        # - "(parameter) query: string"
        type_patterns = [
            # TypeScript/JavaScript: "identifier: Type"
            rf"(?:const|let|var|\(const\)|\(let\)|\(parameter\))\s+{re.escape(var_name)}\s*:\s*([A-Z][a-zA-Z0-9_<>\[\]|&\s]+)",
            # Just ": Type" after identifier
            rf"{re.escape(var_name)}\s*:\s*([A-Z][a-zA-Z0-9_<>\[\]|&\s]+)",
            # Generic type extraction
            r":\s*([A-Z][a-zA-Z0-9_]+)(?:<[^>]+>)?(?:\[\])?",
        ]

        for pattern in type_patterns:
            match = re.search(pattern, hover_contents)
            if match:
                type_str = match.group(1).strip()
                # Clean up: remove generic params and array notation for simpler label
                simple_type = re.match(r"([A-Z][a-zA-Z0-9_]+)", type_str)
                if simple_type:
                    return simple_type.group(1)

        return None

    async def extract_for_file(
        self,
        file_path: Path,
        positions: list[tuple[int, int]] | None = None,
    ) -> dict[tuple[int, int], ContextBlock]:
        """Extract context for multiple positions in a file.

        Args:
            file_path: Path to the source file
            positions: List of (line, character) positions. If None, uses
                      strategic positions (after dots, at line starts)

        Returns:
            Dictionary mapping positions to context blocks
        """
        if positions is None:
            positions = self._find_strategic_positions(file_path)

        results = {}
        for line, char in positions:
            context = await self.extract_at_position(file_path, line, char)
            results[(line, char)] = context

        return results

    def _is_library_internal(self, name: str) -> bool:
        """Check if a symbol name matches library-internal patterns.

        These patterns indicate symbols that are not project-specific,
        such as Zod internal types ($ZodError), private members (_internal),
        or namespaced globals (NodeJS.Process).

        Args:
            name: Symbol name to check

        Returns:
            True if the name matches a library-internal pattern
        """
        for pattern in LIBRARY_INTERNAL_PATTERNS:
            if re.match(pattern, name):
                return True
        return False

    def _extract_methods(
        self, completions: list[CompletionItem]
    ) -> list[MethodSignature]:
        """Extract method signatures from completions.

        Parses LSP completion items to extract full signatures including
        parameters and return types.

        Filters out:
        - Global builtin functions (queueMicrotask, setTimeout, etc.)
        - Library-internal methods ($zodMethod, _privateMethod, etc.)
        """
        methods = []
        seen_names = set()

        for item in completions:
            if item.kind in (
                CompletionItemKind.Method,
                CompletionItemKind.Function,
            ):
                # Skip duplicates
                if item.label in seen_names:
                    continue

                # Filter out global builtins to prevent hallucinations
                if item.label in GLOBAL_BUILTINS_FILTER:
                    continue

                # Filter out library-internal symbols
                if self._is_library_internal(item.label):
                    continue

                seen_names.add(item.label)

                method = self._parse_method_signature(item)
                methods.append(method)

        return methods

    def _parse_method_signature(self, item: CompletionItem) -> MethodSignature:
        """Parse a CompletionItem into a MethodSignature.

        Handles various detail formats from different language servers:
        - TypeScript: "(param: Type): ReturnType"
        - Python: "(param: Type) -> ReturnType"
        """
        name = item.label
        parameters = ""
        return_type = ""
        documentation = item.documentation or ""

        if item.detail:
            detail = item.detail.strip()

            # Try to parse TypeScript style: "(param: Type): ReturnType"
            ts_match = re.match(
                r'\(([^)]*)\)\s*:\s*(.+)',
                detail
            )
            if ts_match:
                parameters = ts_match.group(1).strip()
                return_type = ts_match.group(2).strip()
            else:
                # Try Python style: "(param: Type) -> ReturnType"
                py_match = re.match(
                    r'\(([^)]*)\)\s*->\s*(.+)',
                    detail
                )
                if py_match:
                    parameters = py_match.group(1).strip()
                    return_type = py_match.group(2).strip()
                else:
                    # Just parameters or full signature in label
                    if detail.startswith('('):
                        # Detail is just parameters
                        paren_match = re.match(r'\(([^)]*)\)', detail)
                        if paren_match:
                            parameters = paren_match.group(1).strip()
                            # Check for return type after
                            rest = detail[paren_match.end():].strip()
                            if rest.startswith(':'):
                                return_type = rest[1:].strip()
                            elif rest.startswith('->'):
                                return_type = rest[2:].strip()
                    else:
                        # Detail might be return type or description
                        # Check if it looks like a type
                        if self._looks_like_type(detail):
                            return_type = detail

        return MethodSignature(
            name=name,
            parameters=parameters,
            return_type=return_type,
            documentation=documentation,
        )

    def _looks_like_type(self, text: str) -> bool:
        """Check if text looks like a type annotation."""
        # Simple heuristics for type detection
        type_patterns = [
            r'^[A-Z][a-zA-Z0-9_]*$',  # PascalCase
            r'^[a-z]+\[[^\]]+\]$',  # Generic like list[str]
            r'^Promise<.+>$',  # Promise<T>
            r'^Array<.+>$',  # Array<T>
            r'^void$',
            r'^string$',
            r'^number$',
            r'^boolean$',
            r'^any$',
        ]
        return any(re.match(p, text) for p in type_patterns)

    def _extract_types(
        self, completions: list[CompletionItem]
    ) -> list[TypeInfo]:
        """Extract type information from completions.

        Filters out:
        - Global/standard library types (Array, Promise, Error, etc.)
        - Library-internal types ($ZodAsyncError, _InternalType, etc.)
        """
        types = []
        seen_names = set()

        for item in completions:
            if item.kind in (
                CompletionItemKind.Class,
                CompletionItemKind.Interface,
                CompletionItemKind.Struct,
                CompletionItemKind.Enum,
                CompletionItemKind.TypeParameter,
            ):
                if item.label in seen_names:
                    continue

                # Filter out global/standard library types
                if item.label in GLOBAL_TYPES_FILTER:
                    continue

                # Filter out library-internal types
                if self._is_library_internal(item.label):
                    continue

                seen_names.add(item.label)

                # Map kind to string
                kind_map = {
                    CompletionItemKind.Class: "class",
                    CompletionItemKind.Interface: "interface",
                    CompletionItemKind.Struct: "struct",
                    CompletionItemKind.Enum: "enum",
                    CompletionItemKind.TypeParameter: "type",
                }

                types.append(TypeInfo(
                    name=item.label,
                    kind=kind_map.get(item.kind, "type"),
                ))

        return types

    def _extract_imports(self, symbols: list[SymbolInfo]) -> list[str]:
        """Extract importable module names from symbols."""
        imports = []
        for symbol in symbols:
            if symbol.kind in (
                SymbolKind.Module,
                SymbolKind.Class,
                SymbolKind.Interface,
                SymbolKind.Function,
            ):
                # Use container name if available for qualified name
                if symbol.container_name:
                    imports.append(f"{symbol.container_name}.{symbol.name}")
                else:
                    imports.append(symbol.name)
        return list(set(imports))[:20]  # Dedupe and limit

    def _find_strategic_positions(
        self, file_path: Path
    ) -> list[tuple[int, int]]:
        """Find strategic positions for context extraction.

        Strategic positions are places where completions are most useful:
        - After dots (method access)
        - After opening parentheses (function arguments)
        - At variable assignments
        """
        positions = []
        try:
            content = file_path.read_text()
            lines = content.split("\n")

            for line_num, line in enumerate(lines):
                # Find dots followed by partial identifiers
                for i, char in enumerate(line):
                    if char == ".":
                        # Position after the dot
                        positions.append((line_num, i + 1))
                    elif char == "(":
                        # Position after opening paren (for argument types)
                        positions.append((line_num, i + 1))
                    elif char == "=" and i > 0 and line[i - 1] != "=":
                        # After assignment (for type inference)
                        if i + 2 < len(line):
                            positions.append((line_num, i + 2))

        except Exception as e:
            logger.warning(f"Failed to find strategic positions in {file_path}: {e}")

        # Limit to avoid excessive LSP calls
        return positions[:50]

    async def clear_cache(self) -> None:
        """Clear cached workspace symbols."""
        self._workspace_symbols_cache = None


async def create_context_extractor(
    project_root: Path,
    language: str = "typescript",
    schema_path: Path | None = None,
) -> ContextExtractor:
    """Factory function to create a context extractor with LSP client.

    Args:
        project_root: Root directory of the project
        language: Programming language (typescript, python, etc.)
        schema_path: Optional path to schema.yaml

    Returns:
        Configured ContextExtractor with active LSP connection
    """
    client = LSPClient(language=language, project_root=project_root)
    await client.start()
    return ContextExtractor(client, schema_path=schema_path)


async def extract_batch_context(
    project_root: Path,
    files: list[Path],
    language: str = "typescript",
    schema_path: Path | None = None,
    positions_per_file: int = 5,
) -> dict[Path, dict[tuple[int, int], ContextBlock]]:
    """Extract context for multiple files with connection reuse.

    Optimized batch processing that:
    - Reuses single LSP connection for all files
    - Caches workspace symbols
    - Limits positions per file for efficiency

    Law compliance:
    - L-batch-efficiency: Single connection, caching, limited positions

    Args:
        project_root: Root directory of the project
        files: List of files to process
        language: Programming language
        schema_path: Optional schema file
        positions_per_file: Max positions to extract per file

    Returns:
        Dictionary mapping file paths to their context extractions
    """
    results: dict[Path, dict[tuple[int, int], ContextBlock]] = {}
    extractor = None

    try:
        extractor = await create_context_extractor(
            project_root=project_root,
            language=language,
            schema_path=schema_path,
        )

        for file_path in files:
            try:
                # Get strategic positions
                positions = extractor._find_strategic_positions(file_path)
                # Limit positions for efficiency
                positions = positions[:positions_per_file]

                if positions:
                    file_contexts = await extractor.extract_for_file(
                        file_path, positions
                    )
                    results[file_path] = file_contexts
                else:
                    results[file_path] = {}

            except Exception as e:
                logger.warning(f"Failed to extract context for {file_path}: {e}")
                results[file_path] = {}

    except Exception as e:
        logger.error(f"Batch context extraction failed: {e}")
    finally:
        if extractor:
            await extractor.lsp.stop()

    return results
