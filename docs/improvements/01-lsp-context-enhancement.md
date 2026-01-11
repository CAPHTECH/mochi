# Improvement 1: LSP Context Enhancement

## Overview

LSPから取得するコンテキスト情報を強化し、メソッドのフルシグネチャ（パラメータ名、型、戻り値の型）を
学習データに含めるようにしました。

## Problem

以前の実装では、メソッド名のみまたは切り詰められた詳細情報（50文字制限）しか含まれていませんでした：

```
// Available methods: all, run, prepare
```

これでは、モデルが正しいAPIの使い方（引数の型や戻り値の型）を学習するのに不十分でした。

## Solution

### 新しいデータ構造

#### MethodSignature

メソッドのフルシグネチャを保持する構造体：

```python
@dataclass
class MethodSignature:
    name: str
    parameters: str = ""      # e.g., "sql: string, params?: any[]"
    return_type: str = ""     # e.g., "Promise<T[]>"
    documentation: str = ""   # JSDoc等のドキュメント
```

#### TypeInfo

型情報を保持する構造体：

```python
@dataclass
class TypeInfo:
    name: str
    kind: str = "type"  # type, interface, class, enum
    members: list[str] = field(default_factory=list)
```

### 新しいContext Block形式

**詳細形式（detailed=True）:**

```
// Available methods:
//   all<T>(sql: string): Promise<T[]>
//   run(sql: string, params?: any[]): Promise<void>
//   prepare(sql: string): Statement
// Available types: DuckDBClient, QueryResult
// DB schema: symbols(id, name, kind), files(path, hash)
```

**コンパクト形式（detailed=False）:**

```
// Available methods: all, run, prepare
// Available types: DuckDBClient, QueryResult
```

## Implementation Details

### 1. シグネチャパーシング

`_parse_method_signature()` メソッドが以下の形式を解析：

- **TypeScript**: `(param: Type): ReturnType`
- **Python**: `(param: Type) -> ReturnType`

```python
def _parse_method_signature(self, item: CompletionItem) -> MethodSignature:
    # TypeScript style: "(sql: string): Promise<T>"
    ts_match = re.match(r'\(([^)]*)\)\s*:\s*(.+)', detail)

    # Python style: "(sql: str) -> list[str]"
    py_match = re.match(r'\(([^)]*)\)\s*->\s*(.+)', detail)
```

### 2. 型検出ヒューリスティクス

`_looks_like_type()` で戻り値の型を識別：

- PascalCase（クラス名）
- ジェネリクス（`Promise<T>`, `Array<T>`, `list[T]`）
- プリミティブ型（`string`, `number`, `void` 等）

## Usage

### 学習データ生成

```python
from mochi.lsp import create_context_extractor

async def generate_training_data():
    extractor = await create_context_extractor(
        project_root=Path("/path/to/project"),
        language="typescript"
    )

    context = await extractor.extract_at_position(
        file_path=Path("src/db.ts"),
        line=10,
        character=15
    )

    # 詳細形式でフォーマット
    context_str = context.format(detailed=True)
```

### ContextBlock の作成

```python
from mochi.lsp import MethodSignature, TypeInfo, ContextBlock

methods = [
    MethodSignature(
        name="all",
        parameters="sql: string",
        return_type="Promise<T[]>"
    ),
    MethodSignature(
        name="run",
        parameters="sql: string",
        return_type="Promise<void>"
    ),
]

types = [TypeInfo(name="DuckDBClient", kind="class")]

ctx = ContextBlock(methods=methods, types=types)
print(ctx.format(detailed=True))
```

## Files Changed

- `src/mochi/lsp/context_extractor.py` - MethodSignature, TypeInfo追加、パーサー強化
- `src/mochi/lsp/__init__.py` - 新しい型のエクスポート

## Expected Impact

- **API精度向上**: 正しい引数の型と戻り値の型を学習できる
- **補完品質向上**: メソッドの使い方がより正確になる
- **汎用性**: TypeScript/Python両方に対応

## Next Steps

1. 学習データの再生成（`scripts/regenerate_training_data.py`）
2. モデルの再学習
3. 精度の検証
