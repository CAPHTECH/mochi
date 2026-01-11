# LSP-based Context Extraction for Training Data

## Overview

mochiの学習データ生成において、LSP (Language Server Protocol) を活用して正確な型情報・API情報をコンテキストとして付与する設計。

### 解決する問題

現在のmochiは以下の問題を持つ：
- スキーマ名・API名が不正確（例: `blob_content` → 実際は `blob.content`）
- メソッド名の誤り（例: `db.one()` → 実際は `db.all()[0]`）
- 存在しないテーブル/関数の生成

### 原因

学習データがコード断片のみで構成され、利用可能なAPI・型情報が含まれていない。

### 解決策

LSPを使用して、コード補完位置で実際に利用可能な情報を抽出し、学習データのコンテキストとして付与する。

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Data Generator                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │ Source Code │───▶│ LSP Client  │───▶│ Context Extractor   │  │
│  │   Files     │    │             │    │                     │  │
│  └─────────────┘    └──────┬──────┘    └──────────┬──────────┘  │
│                            │                      │              │
│                            ▼                      ▼              │
│                     ┌─────────────┐    ┌─────────────────────┐  │
│                     │ LSP Server  │    │ Training Example    │  │
│                     │ (tsserver)  │    │ Generator           │  │
│                     └─────────────┘    └─────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## LSP Requests Used

### 1. textDocument/completion

カーソル位置で利用可能な補完候補を取得。

```json
// Request
{
  "method": "textDocument/completion",
  "params": {
    "textDocument": { "uri": "file:///path/to/file.ts" },
    "position": { "line": 10, "character": 15 }
  }
}

// Response
{
  "items": [
    { "label": "all", "kind": 2, "detail": "<T>(sql: string, params?: any[]) => Promise<T[]>" },
    { "label": "run", "kind": 2, "detail": "(sql: string, params?: any[]) => Promise<void>" }
  ]
}
```

### 2. textDocument/hover

シンボルの型情報・ドキュメントを取得。

```json
// Request
{
  "method": "textDocument/hover",
  "params": {
    "textDocument": { "uri": "file:///path/to/file.ts" },
    "position": { "line": 5, "character": 10 }
  }
}

// Response
{
  "contents": {
    "kind": "markdown",
    "value": "```typescript\nclass DuckDBClient\n```\nDuckDB database client."
  }
}
```

### 3. workspace/symbol

プロジェクト全体のシンボル一覧を取得。

```json
// Request
{
  "method": "workspace/symbol",
  "params": { "query": "" }
}

// Response
{
  "result": [
    { "name": "DuckDBClient", "kind": 5, "location": { ... } },
    { "name": "scanFiles", "kind": 12, "location": { ... } }
  ]
}
```

## Training Data Format

### Before (Current)

```json
{
  "text": "### Instruction:\nFill in the typescript code\n\n### Input:\n// File: src/query.ts\nconst result = await db.\n\n### Response:\nall(`SELECT * FROM blob`)"
}
```

### After (With LSP Context)

```json
{
  "text": "### Instruction:\nFill in the typescript code\n\n### Context:\n// Available methods on db: all<T>(sql, params), run(sql, params), exec(sql)\n// Available tables: blob(hash, content, size), tree(commit, path, blob_hash)\n// Imported types: Blob, Tree, Symbol\n\n### Input:\n// File: src/query.ts\nconst result = await db.\n\n### Response:\nall<Blob>(`SELECT * FROM blob WHERE hash = ?`, [hash])"
}
```

## Context Block Structure

```
### Context:
// Available methods: {LSPから取得したメソッド一覧}
// Available types: {スコープ内の型定義}
// DB schema: {プロジェクトのスキーマ情報}
// Imports: {利用可能なモジュール}
```

## Implementation Plan

### Phase 1: LSP Client Implementation

```python
# src/mochi/data/lsp_client.py

class LSPClient:
    """Language Server Protocol client for context extraction."""

    def __init__(self, language: str, project_root: Path):
        self.language = language
        self.project_root = project_root
        self.server_process = None

    def start(self) -> None:
        """Start the LSP server process."""

    def stop(self) -> None:
        """Stop the LSP server process."""

    def get_completions(self, file: Path, line: int, col: int) -> list[CompletionItem]:
        """Get completion items at position."""

    def get_hover(self, file: Path, line: int, col: int) -> HoverInfo:
        """Get hover information at position."""

    def get_workspace_symbols(self) -> list[SymbolInfo]:
        """Get all symbols in workspace."""
```

### Phase 2: Context Extractor

```python
# src/mochi/data/context_extractor.py

class ContextExtractor:
    """Extract context information using LSP."""

    def __init__(self, lsp_client: LSPClient, schema_source: Path | None = None):
        self.lsp = lsp_client
        self.schema = self._load_schema(schema_source)

    def extract_at_position(self, file: Path, line: int, col: int) -> str:
        """Generate context block for a specific position."""
        completions = self.lsp.get_completions(file, line, col)
        types = self._get_relevant_types(file)

        return self._format_context_block(completions, types, self.schema)

    def _format_context_block(self, completions, types, schema) -> str:
        """Format extracted information as context block."""
        lines = []
        if completions:
            methods = [c.label for c in completions if c.kind == CompletionKind.Method]
            lines.append(f"// Available methods: {', '.join(methods[:10])}")
        if types:
            lines.append(f"// Available types: {', '.join(types[:10])}")
        if schema:
            lines.append(f"// DB schema: {schema}")
        return '\n'.join(lines)
```

### Phase 3: Training Data Generator Integration

```python
# src/mochi/data/generator.py (modified)

class TrainingDataGenerator:
    def __init__(self, project_root: Path, language: str = "typescript"):
        self.lsp_client = LSPClient(language, project_root)
        self.context_extractor = ContextExtractor(self.lsp_client)

    def generate_example(self, file: Path, split_point: int) -> dict:
        """Generate a training example with LSP context."""
        content = file.read_text()
        line, col = self._offset_to_position(content, split_point)

        context = self.context_extractor.extract_at_position(file, line, col)
        prefix = content[:split_point]
        suffix = content[split_point:]

        return {
            "text": f"### Instruction:\nFill in the typescript code\n\n### Context:\n{context}\n\n### Input:\n{prefix}\n\n### Response:\n{suffix[:100]}"
        }
```

## Language Support

| Language | LSP Server | Package |
|----------|------------|---------|
| TypeScript/JavaScript | tsserver | typescript |
| Python | pylsp | python-lsp-server |
| Rust | rust-analyzer | rust-analyzer |
| Go | gopls | gopls |
| Java | jdtls | eclipse.jdt.ls |

## Schema Extraction

DBスキーマは以下のソースから抽出：

1. **Migration files**: SQL migration から CREATE TABLE を解析
2. **ORM models**: Prisma schema, TypeORM entities など
3. **Schema definition files**: schema.ts, schema.sql など
4. **Manual configuration**: schema.yaml で明示指定

```yaml
# schema.yaml (manual override)
tables:
  - name: blob
    columns: [hash, content, size]
  - name: tree
    columns: [commit, path, blob_hash]
  - name: symbol
    columns: [blob_hash, name, kind, signature, doc]
```

## Performance Considerations

### Batch Processing

LSPサーバーの起動コストを分散するため、バッチ処理を採用：

```python
async def generate_batch(files: list[Path]) -> list[dict]:
    async with LSPClient("typescript", project_root) as lsp:
        examples = []
        for file in files:
            # LSP connection reused
            example = await generate_example(file, lsp)
            examples.append(example)
    return examples
```

### Caching

頻繁に参照される情報をキャッシュ：

```python
@lru_cache(maxsize=1000)
def get_workspace_symbols(project_root: str) -> list[SymbolInfo]:
    ...

@lru_cache(maxsize=100)
def get_file_types(file_path: str) -> list[str]:
    ...
```

## Expected Improvements

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Schema name accuracy | ~30% | ~90% |
| Method name accuracy | ~50% | ~95% |
| Type annotation accuracy | ~60% | ~90% |
| Overall code correctness | ~70% | ~85% |

## Future Extensions

1. **Multi-language support**: 複数言語のLSPを統合
2. **Incremental updates**: ファイル変更時の差分更新
3. **Custom LSP extensions**: プロジェクト固有の情報取得
4. **Semantic context**: 関数の意味的な説明を付与

## References

- [Language Server Protocol Specification](https://microsoft.github.io/language-server-protocol/)
- [typescript-language-server](https://github.com/typescript-language-server/typescript-language-server)
- [python-lsp-server](https://github.com/python-lsp/python-lsp-server)
