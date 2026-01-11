# Improvement 5: Evaluation Pipeline

## Overview

複数のモデルとタスク種別に対応した包括的な評価パイプラインを追加しました。
CI統合のためのJSON出力とタスク種別ごとのメトリクスを提供します。

## Problem

以前の実装では：
- 単一モデル（Qwen2.5-Coder-1.5B）のみ対応
- タスク種別の区別なし
- 手動でのスクリプト実行のみ
- 結果がテキスト出力のみ（CI統合困難）

## Solution

### 新しい評価スクリプト

`scripts/evaluate_model.py` - 包括的な評価パイプライン

#### 特徴

1. **複数モデル対応**: `--preset` オプションでモデル選択
2. **タスク種別フィルタ**: `--task-type` で特定タスクのみ評価
3. **JSON出力**: `--output-json` でCI統合用レポート生成
4. **構造化レポート**: タスク種別ごとのメトリクス

### テストケース構造

```python
@dataclass
class TestCase:
    name: str              # テスト名
    task_type: str         # completion, method_call, etc.
    instruction: str       # 指示文
    input_text: str        # 入力コード
    context: str           # LSPコンテキスト
    expected_keywords: list[str]  # 期待されるキーワード
    description: str       # テストの説明
    weight: float = 1.0    # スコア計算用重み
```

### タスク種別別テストケース

| TaskType | テスト数 | 説明 |
|----------|---------|------|
| method_call | 4 | メソッド呼び出し補完（db.all, expect.toBe等） |
| completion | 2 | 一般コード補完 |
| import | 1 | インポート文生成 |
| documentation | 1 | JSDoc生成 |
| explanation | 1 | コード説明 |

### 評価レポート構造

```python
@dataclass
class EvaluationReport:
    model_preset: str
    adapter_path: str
    timestamp: str
    total_tests: int
    passed_tests: int
    overall_accuracy: float
    by_task_type: dict[str, dict]  # タスク種別ごとのメトリクス
    results: list[TestResult]
    total_time_seconds: float
```

### JSON出力例

```json
{
  "model_preset": "qwen3-coder",
  "adapter_path": "/path/to/adapter",
  "timestamp": "2025-01-11T12:00:00",
  "total_tests": 9,
  "passed_tests": 7,
  "overall_accuracy": 78.5,
  "by_task_type": {
    "method_call": {
      "total": 4,
      "passed": 3,
      "accuracy": 75.0,
      "avg_time_ms": 850.0
    },
    "completion": {
      "total": 2,
      "passed": 2,
      "accuracy": 100.0,
      "avg_time_ms": 1200.0
    }
  },
  "results": [...]
}
```

## Usage

### 基本的な使用

```bash
# デフォルト（qwen3-coder）で評価
python scripts/evaluate_model.py

# GPT-OSSモデルで評価
python scripts/evaluate_model.py --preset gpt-oss

# 特定のタスク種別のみ評価
python scripts/evaluate_model.py --task-type method_call

# JSON出力
python scripts/evaluate_model.py --output-json results.json

# 簡易出力（サマリのみ）
python scripts/evaluate_model.py --quiet
```

### CI統合

```yaml
# GitHub Actions example
- name: Evaluate model
  run: |
    python scripts/evaluate_model.py \
      --preset qwen3-coder \
      --output-json evaluation-results.json

- name: Upload results
  uses: actions/upload-artifact@v4
  with:
    name: evaluation-results
    path: evaluation-results.json
```

## InferenceConfig統合

評価時は`InferenceConfig.for_task()`を使用して、タスク種別ごとの最適設定を適用：

```python
# 評価ループ内
task_type_enum = TaskType(test.task_type)
config = InferenceConfig.for_task(task_type_enum)

result = engine.generate_with_config(
    instruction=test.instruction,
    input_text=test.input_text,
    context=test.context,
    task_type=task_type_enum,
    config=config,
)
```

これにより、評価結果がタスク種別ごとの最適設定を反映します。

## 評価基準

### キーワードマッチング

```python
found_keywords = [
    kw for kw in test.expected_keywords
    if kw.lower() in result.response.lower()
]
accuracy = len(found_keywords) / len(test.expected_keywords) * 100
```

- 大文字小文字を無視
- 部分一致（キーワードがレスポンスに含まれるか）
- 精度50%以上で合格

### タスク種別別メトリクス

各タスク種別ごとに以下を計算：
- `total`: テスト総数
- `passed`: 合格数
- `accuracy`: 平均精度
- `avg_time_ms`: 平均推論時間

## Files Changed

- `scripts/evaluate_model.py` (NEW)
  - `TestCase` データクラス
  - `TestResult` データクラス
  - `EvaluationReport` データクラス
  - `evaluate_model()` 関数
  - `print_summary()` 関数
  - `save_json_report()` 関数
  - CLI引数処理

## Expected Impact

- **CI統合**: JSON出力でGitHub Actions等との連携
- **モデル比較**: 複数モデルの比較評価が容易に
- **タスク別分析**: 弱点のあるタスク種別を特定可能
- **自動検証**: PRマージ前の品質ゲート

## Backward Compatibility

既存の `verify_accuracy.py` は変更なし。新しいスクリプトは追加機能として提供。

## Next Steps

1. テストケースの拡充（より多様なシナリオ）
2. ベースラインモデル（adapter無し）との比較機能
3. 経時的なメトリクス追跡（過去の評価結果との比較）
