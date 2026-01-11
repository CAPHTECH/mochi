# Improvement 3: Prompt Design Optimization

## Overview

タスク種別に応じた最適なプロンプト形式を使用し、LSPコンテキストを含むプロンプトテンプレートを追加しました。

## Problem

以前の実装では、すべてのタスクに同じAlpacaスタイルのプロンプトを使用していました。
また、LSPコンテキスト（利用可能なメソッド、型情報）をプロンプトに含める方法がありませんでした。

## Solution

### TaskType Enum

タスクの種類を明示的に指定できるようになりました：

```python
class TaskType(Enum):
    COMPLETION = "completion"      # コード補完
    METHOD_CALL = "method_call"    # メソッド呼び出し補完
    EXPLANATION = "explanation"    # コード説明
    DOCUMENTATION = "documentation" # ドキュメント生成
    IMPORT = "import"              # インポート文追加
    GENERAL = "general"            # 汎用
```

### PromptTemplate クラス

複数のプロンプト形式をサポート：

#### 1. Alpaca with Input（標準形式）
```
### Instruction:
{instruction}

### Input:
{input}

### Response:
```

#### 2. Context-Aware（LSPコンテキスト付き）
```
### Instruction:
{instruction}

### Context:
{context}

### Input:
{input}

### Response:
```

#### 3. Minimal（コード補完用）
```
{context}
{code}
```

### 使用例

```python
from mochi.mcp.inference_mlx import TaskType, PromptTemplate

# Context-aware プロンプトの生成
prompt = PromptTemplate.format(
    task_type=TaskType.METHOD_CALL,
    instruction="Complete the method call:",
    input_text="const result = await db.",
    context="// Available methods:\n//   all(sql: string): Promise<T[]>",
)
```

### Inference Engine の更新

`generate()` メソッドに新しいパラメータを追加：

```python
def generate(
    self,
    instruction: str,
    input_text: str = "",
    context: str = "",                    # NEW: LSPコンテキスト
    task_type: TaskType = TaskType.COMPLETION,  # NEW: タスク種別
    max_new_tokens: int = 2048,
    temperature: float = 0.1,
    top_p: float = 0.5,
    use_alpaca_format: bool = True,       # NEW: フォーマット選択
) -> InferenceResult:
```

### 便利メソッドの追加

#### `generate_completion()`
コード補完に最適化された設定：
- 短い出力（256トークン）
- 低いtemperature（0.1）

```python
result = engine.generate_completion(
    code_prefix="const users = await db.",
    context="// Available methods:\n//   all(): Promise<T[]>",
)
```

#### `generate_method_completion()`
メソッド呼び出し補完に最適化：
- さらに短い出力（128トークン）
- 非常に低いtemperature（0.05）で決定的な出力

```python
result = engine.generate_method_completion(
    code_with_dot="db.",
    available_methods="//   all(sql: string): Promise<T[]>\n//   run(sql: string): void",
)
```

## Implementation Details

### PromptTemplate.format() のロジック

1. `use_alpaca=False` の場合：ミニマル形式を使用
2. `context` が提供された場合：Context-Aware形式を使用
3. `input_text` が提供された場合：Alpaca with Input形式を使用
4. それ以外：Alpaca without Input形式を使用

### デフォルトInstruction

タスク種別ごとにデフォルトのinstructionを自動設定：

```python
instructions = {
    TaskType.COMPLETION: "Complete the following code:",
    TaskType.METHOD_CALL: "Complete the method call:",
    TaskType.EXPLANATION: "Explain what this code does:",
    TaskType.DOCUMENTATION: "Add documentation to this code:",
    TaskType.IMPORT: "Add the necessary imports:",
    TaskType.GENERAL: "Complete the following:",
}
```

## Files Changed

- `src/mochi/mcp/inference_mlx.py`
  - `TaskType` enum追加
  - `PromptTemplate` クラス追加
  - `generate()` メソッド更新
  - `generate_completion()` メソッド追加
  - `generate_method_completion()` メソッド追加

## Expected Impact

- **精度向上**: タスクに最適化されたプロンプト
- **API補完改善**: LSPコンテキストを含むことでメソッド名の精度向上
- **柔軟性**: 用途に応じたプロンプト形式の選択

## Backward Compatibility

既存の `generate()` 呼び出しは引き続き動作します（新しいパラメータはすべてオプショナル）。
