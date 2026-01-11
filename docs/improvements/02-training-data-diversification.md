# Improvement 2: Training Data Diversification

## Overview

学習データに多様なinstruction種別を追加し、モデルがより多くのタスクを学習できるようにしました。

## Problem

以前の実装では、以下の2種類のタスクしかありませんでした：
1. コード補完（Fill in the code）
2. コード説明（Explain the code）

これでは、モデルがメソッド呼び出しのパターンやAPI使用法を十分に学習できませんでした。

## Solution

### 新しいInstruction種別

#### 1. Method Call Completion (API学習に重要)

メソッド呼び出しの補完に特化した例を生成します。

```
Instruction: Complete the method call in this typescript code:
Input:
// File: src/db.ts
// Available methods:
//   all(sql: string): Promise<T[]>
//   run(sql: string): Promise<void>

const db = new DuckDB();
const result = await db.

Output: all("SELECT * FROM users")
```

#### 2. Documentation Generation

コードからドキュメントを生成する例を追加：

```
Instruction: Add documentation comments to this typescript code:
Input:
async function fetchUser(id: number): Promise<User> {
    return await db.all("SELECT * FROM users WHERE id = ?", [id]);
}

Output:
Fetches a user by their ID from the database.
@param id - The user's unique identifier
@returns Promise resolving to the User object
```

#### 3. Import Statement Completion

必要なインポート文を推測する例：

```
Instruction: Add the necessary import statements for this typescript code:
Input:
// File: src/handler.ts
export async function handle(req: Request): Promise<Response> {
    const db = new DuckDBClient();
    const users = await db.all("SELECT * FROM users");
    return new Response(JSON.stringify(users));
}

Output:
import { DuckDBClient } from "./shared/duckdb.js";
```

### 追加のテンプレート（将来の拡張用）

以下のテンプレートも定義されていますが、現在は未使用です：

- **Refactoring**: コードの改善
- **Type Annotation**: 型注釈の追加
- **Error Handling**: エラーハンドリングの追加

## Implementation Details

### AlpacaConverterの変更

```python
class AlpacaConverter:
    # 新しいテンプレート
    METHOD_CALL_TEMPLATES = [
        "Complete the method call in this {language} code:",
        "What method should be called here?",
        "Fill in the appropriate API call:",
    ]

    DOCSTRING_TEMPLATES = [
        "Add documentation comments to this {language} code:",
        "Write JSDoc/docstring for this {chunk_type}:",
        "Document this {language} function/class:",
    ]

    IMPORT_TEMPLATES = [
        "Add the necessary import statements for this {language} code:",
        "What imports are needed for this code?",
        "Complete the import statements:",
    ]

    def convert_chunks(
        self,
        chunks: list[CodeChunk],
        include_completion: bool = True,
        include_explanation: bool = True,
        include_method_call: bool = True,  # 新規
        include_docstring: bool = True,    # 新規
        ...
    ) -> list[AlpacaExample]:
```

### 新しいメソッド

1. **`_create_method_call_examples()`**
   - コード内のドット（`.`）を検出
   - ドット以前をinput、ドット以降をoutputとして例を生成
   - LSPコンテキストを含めることで、利用可能なメソッドを明示

2. **`_create_docstring_examples()`**
   - JSDoc/docstringを持つコードチャンクを検出
   - ドキュメントを除いたコードをinput、ドキュメントをoutputとして例を生成

3. **`_create_import_examples()`**
   - import文を持つコードを検出
   - import以外のコードをinput、import文をoutputとして例を生成

## Usage

```python
from mochi.data_generation.alpaca_converter import AlpacaConverter

converter = AlpacaConverter("project")

# すべてのタスク種別を含める（デフォルト）
examples = converter.convert_chunks(chunks)

# 特定のタスク種別のみ
examples = converter.convert_chunks(
    chunks,
    include_completion=True,
    include_explanation=False,
    include_method_call=True,
    include_docstring=False,
)
```

## Files Changed

- `src/mochi/data_generation/alpaca_converter.py`
  - 新しいテンプレート追加
  - `_create_method_call_examples()` 追加
  - `_create_docstring_examples()` 追加
  - `_create_import_examples()` 追加（将来の使用のため）

## Expected Impact

- **API精度向上**: メソッド呼び出しパターンを直接学習
- **コンテキスト理解**: LSPコンテキストと実際の使用例を結びつける
- **多様なタスク**: モデルの汎用性向上

## Next Steps

1. 学習データの再生成
2. 各タスク種別の割合調整（必要に応じて）
3. 精度の検証
