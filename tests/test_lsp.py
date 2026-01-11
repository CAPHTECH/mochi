"""Tests for LSP-based context extraction.

These tests verify the LSP client and context extraction functionality.
Some tests require a TypeScript language server (tsserver) to be available.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mochi.lsp.client import (
    CompletionItem,
    CompletionItemKind,
    HoverInfo,
    LSPClient,
    SymbolInfo,
    SymbolKind,
)
from mochi.lsp.context_extractor import (
    ContextBlock,
    ContextExtractor,
    MethodSignature,
    SchemaInfo,
    TypeInfo,
)


class TestMethodSignature:
    """Tests for MethodSignature formatting."""

    def test_format_short_with_params_and_return(self):
        """MethodSignature formats with parameters and return type."""
        sig = MethodSignature(
            name="all",
            parameters="sql: string",
            return_type="Promise<T[]>"
        )
        assert sig.format_short() == "all(sql: string): Promise<T[]>"

    def test_format_short_name_only(self):
        """MethodSignature formats with name only."""
        sig = MethodSignature(name="close")
        assert sig.format_short() == "close"

    def test_format_full_with_documentation(self):
        """MethodSignature includes documentation in full format."""
        sig = MethodSignature(
            name="run",
            parameters="sql: string",
            return_type="void",
            documentation="Execute SQL statement."
        )
        full = sig.format_full()
        assert "run(sql: string): void" in full
        assert "Execute SQL statement" in full


class TestTypeInfo:
    """Tests for TypeInfo formatting."""

    def test_format_simple(self):
        """TypeInfo formats simple type."""
        t = TypeInfo(name="DuckDBClient", kind="class")
        assert t.format() == "DuckDBClient"

    def test_format_with_members(self):
        """TypeInfo formats type with members."""
        t = TypeInfo(
            name="User",
            kind="interface",
            members=["id", "name", "email"]
        )
        result = t.format()
        assert "User" in result
        assert "id" in result
        assert "name" in result


class TestContextBlock:
    """Tests for ContextBlock formatting."""

    def test_format_detailed_with_methods(self):
        """ContextBlock formats methods in detailed mode."""
        methods = [
            MethodSignature(name="all", parameters="sql: string", return_type="Promise<T[]>"),
            MethodSignature(name="run", parameters="sql: string", return_type="void"),
        ]
        block = ContextBlock(methods=methods)
        result = block.format(detailed=True)
        assert "// Available methods:" in result
        assert "all(sql: string): Promise<T[]>" in result
        assert "run(sql: string): void" in result

    def test_format_compact_with_methods(self):
        """ContextBlock formats methods in compact mode."""
        methods = [
            MethodSignature(name="all", parameters="sql: string"),
            MethodSignature(name="run", parameters="sql: string"),
        ]
        block = ContextBlock(methods=methods)
        result = block.format_compact()
        assert "// Available methods: all, run" in result

    def test_format_with_method_names(self):
        """ContextBlock works with legacy method_names."""
        block = ContextBlock(method_names=["all", "run", "exec"])
        result = block.format(detailed=False)
        assert "// Available methods: all, run, exec" in result

    def test_format_with_types(self):
        """ContextBlock formats types correctly."""
        types = [
            TypeInfo(name="Blob", kind="class"),
            TypeInfo(name="Tree", kind="interface"),
        ]
        block = ContextBlock(types=types)
        result = block.format()
        assert "// Available types: Blob, Tree" in result

    def test_format_with_type_names(self):
        """ContextBlock works with legacy type_names."""
        block = ContextBlock(type_names=["Blob", "Tree", "Symbol"])
        result = block.format()
        assert "// Available types: Blob, Tree, Symbol" in result

    def test_format_with_schema(self):
        """ContextBlock formats schema correctly."""
        block = ContextBlock(schema="blob(hash, content), tree(commit, path)")
        result = block.format()
        assert "// DB schema: blob(hash, content), tree(commit, path)" in result

    def test_format_with_imports(self):
        """ContextBlock formats imports correctly."""
        block = ContextBlock(imports=["lodash", "express"])
        result = block.format()
        assert "// Imports: lodash, express" in result

    def test_format_full_block_detailed(self):
        """ContextBlock formats complete block in detailed mode."""
        block = ContextBlock(
            methods=[
                MethodSignature(name="get", parameters="key: string", return_type="T"),
                MethodSignature(name="set", parameters="key: string, value: T"),
            ],
            types=[TypeInfo(name="User"), TypeInfo(name="Session")],
            schema="users(id, name)",
            imports=["express"],
        )
        result = block.format(detailed=True)
        lines = result.split("\n")
        # Methods header + 2 method lines + types + schema + imports = 6 lines
        assert len(lines) >= 5
        assert "// Available methods:" in result
        assert "get(key: string): T" in result
        assert "// Available types:" in result
        assert "// DB schema:" in result
        assert "// Imports:" in result

    def test_empty_block(self):
        """Empty ContextBlock returns empty string."""
        block = ContextBlock()
        assert block.is_empty()
        assert block.format() == ""

    def test_limits_methods(self):
        """ContextBlock limits number of methods shown."""
        methods = [MethodSignature(name=f"method{i}") for i in range(20)]
        block = ContextBlock(methods=methods)
        result = block.format()
        # Should only show first 15
        assert "method14" in result
        assert "method15" not in result


class TestSchemaInfo:
    """Tests for SchemaInfo loading and formatting."""

    def test_format_empty(self):
        """Empty schema returns empty string."""
        schema = SchemaInfo()
        assert schema.format() == ""

    def test_format_tables(self):
        """SchemaInfo formats tables correctly."""
        schema = SchemaInfo(
            tables=[
                {"name": "blob", "columns": ["hash", "content", "size"]},
                {"name": "tree", "columns": ["commit", "path"]},
            ]
        )
        result = schema.format()
        assert "blob(hash, content, size)" in result
        assert "tree(commit, path)" in result

    def test_from_yaml_missing_file(self):
        """SchemaInfo handles missing file gracefully."""
        schema = SchemaInfo.from_yaml(Path("/nonexistent/schema.yaml"))
        assert schema.tables == []


class TestCompletionItem:
    """Tests for CompletionItem data class."""

    def test_create_method_completion(self):
        """Can create a method completion item."""
        item = CompletionItem(
            label="all",
            kind=CompletionItemKind.Method,
            detail="<T>(sql: string) => Promise<T[]>",
        )
        assert item.label == "all"
        assert item.kind == CompletionItemKind.Method
        assert "Promise" in item.detail

    def test_create_type_completion(self):
        """Can create a type completion item."""
        item = CompletionItem(
            label="Blob",
            kind=CompletionItemKind.Class,
            documentation="Binary blob storage.",
        )
        assert item.label == "Blob"
        assert item.kind == CompletionItemKind.Class


class TestContextExtractor:
    """Tests for ContextExtractor logic (mocked LSP)."""

    @pytest.fixture
    def mock_lsp_client(self):
        """Create a mock LSP client."""
        client = MagicMock(spec=LSPClient)
        client.get_completions = AsyncMock(
            return_value=[
                CompletionItem("all", CompletionItemKind.Method, "(sql) => T[]"),
                CompletionItem("run", CompletionItemKind.Method, "(sql) => void"),
                CompletionItem("Blob", CompletionItemKind.Class, None),
            ]
        )
        client.get_workspace_symbols = AsyncMock(
            return_value=[
                SymbolInfo("DuckDBClient", SymbolKind.Class, "file://test.ts"),
                SymbolInfo("scanFiles", SymbolKind.Function, "file://test.ts"),
            ]
        )
        return client

    @pytest.fixture
    def extractor(self, mock_lsp_client):
        """Create a ContextExtractor with mocked LSP."""
        return ContextExtractor(mock_lsp_client)

    @pytest.mark.asyncio
    async def test_extract_at_position(self, extractor, mock_lsp_client, tmp_path):
        """ContextExtractor extracts context at position."""
        # Create a test file
        test_file = tmp_path / "test.ts"
        test_file.write_text("const db = new DuckDB();\ndb.")

        context = await extractor.extract_at_position(test_file, 1, 3)

        assert not context.is_empty()
        # Methods are MethodSignature objects
        method_names = [m.name for m in context.methods]
        assert "all" in method_names
        assert "run" in method_names
        # Types are TypeInfo objects
        type_names = [t.name for t in context.types]
        assert "Blob" in type_names

    @pytest.mark.asyncio
    async def test_extract_with_schema(self, mock_lsp_client, tmp_path):
        """ContextExtractor includes schema when available."""
        # Create schema file
        schema_path = tmp_path / "schema.yaml"
        schema_path.write_text(
            """
tables:
  - name: blob
    columns: [hash, content]
"""
        )

        extractor = ContextExtractor(mock_lsp_client, schema_path=schema_path)
        test_file = tmp_path / "test.ts"
        test_file.write_text("const x = 1;")

        context = await extractor.extract_at_position(test_file, 0, 0)

        assert "blob(hash, content)" in context.schema

    def test_extract_methods_from_completions(self, extractor):
        """_extract_methods filters to methods only and returns MethodSignature."""
        completions = [
            CompletionItem("all", CompletionItemKind.Method, "(sql: string): Promise<T[]>"),
            CompletionItem("run", CompletionItemKind.Function, "(sql: string): void"),
            CompletionItem("Blob", CompletionItemKind.Class, None),
            CompletionItem("const", CompletionItemKind.Keyword, None),
        ]

        methods = extractor._extract_methods(completions)

        # Returns MethodSignature objects
        method_names = [m.name for m in methods]
        assert "all" in method_names
        assert "run" in method_names
        assert "Blob" not in method_names
        assert "const" not in method_names

        # Check signature parsing
        all_method = next(m for m in methods if m.name == "all")
        assert all_method.parameters == "sql: string"
        assert all_method.return_type == "Promise<T[]>"

    def test_extract_types_from_completions(self, extractor):
        """_extract_types filters to types only and returns TypeInfo."""
        completions = [
            CompletionItem("all", CompletionItemKind.Method, None),
            CompletionItem("Blob", CompletionItemKind.Class, None),
            CompletionItem("IDatabase", CompletionItemKind.Interface, None),
        ]

        types = extractor._extract_types(completions)

        # Returns TypeInfo objects
        type_names = [t.name for t in types]
        assert "Blob" in type_names
        assert "IDatabase" in type_names
        assert "all" not in type_names

        # Check kind is set
        blob_type = next(t for t in types if t.name == "Blob")
        assert blob_type.kind == "class"

    def test_parse_typescript_signature(self, extractor):
        """_parse_method_signature parses TypeScript style signatures."""
        item = CompletionItem(
            label="query",
            kind=CompletionItemKind.Method,
            detail="(sql: string, params?: any[]): Promise<Row[]>",
            documentation="Execute a query.",
        )

        sig = extractor._parse_method_signature(item)

        assert sig.name == "query"
        assert sig.parameters == "sql: string, params?: any[]"
        assert sig.return_type == "Promise<Row[]>"
        assert sig.documentation == "Execute a query."

    def test_parse_python_signature(self, extractor):
        """_parse_method_signature parses Python style signatures."""
        item = CompletionItem(
            label="execute",
            kind=CompletionItemKind.Method,
            detail="(sql: str, params: list) -> list[dict]",
        )

        sig = extractor._parse_method_signature(item)

        assert sig.name == "execute"
        assert sig.parameters == "sql: str, params: list"
        assert sig.return_type == "list[dict]"


class TestLSPClientConfig:
    """Tests for LSP client configuration."""

    def test_supported_languages(self):
        """LSPClient has commands for common languages."""
        client = LSPClient("typescript", Path("/tmp"))
        cmd = client._get_server_command()
        assert "typescript-language-server" in " ".join(cmd)

    def test_unsupported_language_raises(self):
        """LSPClient raises for unsupported languages."""
        client = LSPClient("cobol", Path("/tmp"))
        with pytest.raises(ValueError, match="Unsupported language"):
            client._get_server_command()


# Integration tests (require actual LSP server)
@pytest.mark.integration
class TestLSPIntegration:
    """Integration tests that require a running LSP server.

    Run with: pytest -m integration tests/test_lsp.py
    """

    @pytest.fixture
    def typescript_project(self, tmp_path):
        """Create a minimal TypeScript project for testing."""
        # Create package.json
        (tmp_path / "package.json").write_text(
            '{"name": "test", "dependencies": {"typescript": "^5.0"}}'
        )
        # Create tsconfig.json
        (tmp_path / "tsconfig.json").write_text(
            '{"compilerOptions": {"target": "ES2020", "module": "NodeNext"}}'
        )
        # Create a source file
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "index.ts").write_text(
            """
interface User {
  id: number;
  name: string;
}

const users: User[] = [];

// This is where we'd test completions
users.
"""
        )
        return tmp_path

    @pytest.mark.asyncio
    async def test_real_lsp_completions(self, typescript_project):
        """Test LSP completions with real tsserver."""
        pytest.skip("Requires typescript-language-server installed")

        client = LSPClient("typescript", typescript_project)
        try:
            await client.start()

            file_path = typescript_project / "src" / "index.ts"
            completions = await client.get_completions(file_path, 9, 6)

            # Should get array methods
            labels = [c.label for c in completions]
            assert "push" in labels or "length" in labels or len(completions) > 0

        finally:
            await client.stop()
