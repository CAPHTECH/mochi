"""Code chunker using tree-sitter for AST-based splitting."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import tree_sitter_typescript as ts_typescript
from tree_sitter import Language, Parser


class ChunkStrategy(Enum):
    """Chunking strategy for code splitting."""

    FUNCTION = "function"
    CLASS = "class"
    FILE = "file"
    TOPLEVEL = "toplevel"


@dataclass
class CodeChunk:
    """Represents a chunk of code."""

    source_file: str
    content: str
    chunk_type: str  # function, class, method, etc.
    name: str | None  # function/class name if available
    start_line: int
    end_line: int
    language: str
    context: str | None = None  # surrounding context (imports, class definition, etc.)


class CodeChunker:
    """Chunks code into meaningful units using tree-sitter."""

    def __init__(self) -> None:
        """Initialize the chunker with language parsers."""
        self._parsers: dict[str, Parser] = {}
        self._setup_parsers()

    def _setup_parsers(self) -> None:
        """Set up tree-sitter parsers for supported languages."""
        # TypeScript/TSX
        ts_language = Language(ts_typescript.language_typescript())
        tsx_language = Language(ts_typescript.language_tsx())

        ts_parser = Parser(ts_language)
        tsx_parser = Parser(tsx_language)

        self._parsers["typescript"] = ts_parser
        self._parsers["tsx"] = tsx_parser

    def chunk(
        self,
        source_file: str,
        content: str,
        language: str,
        strategy: ChunkStrategy = ChunkStrategy.TOPLEVEL,
        max_lines: int = 200,
    ) -> list[CodeChunk]:
        """
        Chunk code into meaningful units.

        Args:
            source_file: Path to the source file
            content: Source code content
            language: Programming language
            strategy: Chunking strategy
            max_lines: Maximum lines per chunk (for FILE strategy)
        """
        if strategy == ChunkStrategy.FILE:
            return self._chunk_by_file(source_file, content, language, max_lines)

        parser = self._get_parser(language)
        if parser is None:
            # Fallback to file-based chunking for unsupported languages
            return self._chunk_by_file(source_file, content, language, max_lines)

        tree = parser.parse(content.encode("utf-8"))
        return self._extract_chunks(source_file, content, language, tree.root_node, strategy)

    def _get_parser(self, language: str) -> Parser | None:
        """Get parser for a language."""
        if language in ("typescript", "javascript"):
            return self._parsers.get("typescript")
        return self._parsers.get(language)

    def _extract_chunks(
        self,
        source_file: str,
        content: str,
        language: str,
        root_node: any,
        strategy: ChunkStrategy,
    ) -> list[CodeChunk]:
        """Extract chunks from AST."""
        chunks: list[CodeChunk] = []
        lines = content.split("\n")

        # Extract imports/context at the top
        imports_end = 0
        for child in root_node.children:
            if child.type in ("import_statement", "import_declaration"):
                imports_end = child.end_point[0] + 1
            else:
                break

        context = "\n".join(lines[:imports_end]) if imports_end > 0 else None

        # Target node types based on strategy
        if strategy == ChunkStrategy.FUNCTION:
            target_types = {
                "function_declaration",
                "arrow_function",
                "function_expression",
                "method_definition",
            }
        elif strategy == ChunkStrategy.CLASS:
            target_types = {"class_declaration", "class_expression"}
        else:  # TOPLEVEL
            target_types = {
                "function_declaration",
                "class_declaration",
                "export_statement",
                "lexical_declaration",
                "variable_declaration",
                "type_alias_declaration",
                "interface_declaration",
            }

        self._collect_chunks(
            source_file, content, language, root_node, target_types, context, chunks
        )

        return chunks

    def _collect_chunks(
        self,
        source_file: str,
        content: str,
        language: str,
        node: any,
        target_types: set[str],
        context: str | None,
        chunks: list[CodeChunk],
    ) -> None:
        """Recursively collect chunks from AST nodes."""
        if node.type in target_types:
            name = self._extract_name(node)
            chunk_content = content[node.start_byte : node.end_byte]

            chunks.append(
                CodeChunk(
                    source_file=source_file,
                    content=chunk_content,
                    chunk_type=node.type,
                    name=name,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    language=language,
                    context=context,
                )
            )
        else:
            # Recurse into children
            for child in node.children:
                self._collect_chunks(
                    source_file, content, language, child, target_types, context, chunks
                )

    def _extract_name(self, node: any) -> str | None:
        """Extract the name from a node (function name, class name, etc.)."""
        for child in node.children:
            if child.type in ("identifier", "property_identifier"):
                return child.text.decode("utf-8")
            # For exported declarations, look deeper
            if child.type in ("function_declaration", "class_declaration"):
                return self._extract_name(child)
        return None

    def _chunk_by_file(
        self,
        source_file: str,
        content: str,
        language: str,
        max_lines: int,
    ) -> list[CodeChunk]:
        """Chunk by file with optional line limit."""
        lines = content.split("\n")

        if len(lines) <= max_lines:
            return [
                CodeChunk(
                    source_file=source_file,
                    content=content,
                    chunk_type="file",
                    name=source_file,
                    start_line=1,
                    end_line=len(lines),
                    language=language,
                )
            ]

        # Split into multiple chunks
        chunks: list[CodeChunk] = []
        for i in range(0, len(lines), max_lines):
            chunk_lines = lines[i : i + max_lines]
            chunks.append(
                CodeChunk(
                    source_file=source_file,
                    content="\n".join(chunk_lines),
                    chunk_type="file_part",
                    name=f"{source_file}:{i + 1}-{min(i + max_lines, len(lines))}",
                    start_line=i + 1,
                    end_line=min(i + max_lines, len(lines)),
                    language=language,
                )
            )

        return chunks
