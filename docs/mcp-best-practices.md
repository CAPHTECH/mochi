# Mochi MCP ベストプラクティスガイド

mochiをMCP経由で効果的に活用するためのガイドです。

## 1. 基本的な使い方

### 1.1 domain_query ツール（推奨）

コード補完の主要ツール。instruction-input形式でコード生成を行います。

```typescript
mcp__mochi__domain_query({
  instruction: "Fill in the typescript code",
  input: "// File: src/indexer/scanner.ts\nfunction scanDirectory(",
  context: "// Available methods...",  // オプション
  validate: true,                       // オプション
  mode: "auto"                          // オプション
})
```

**パラメータ:**
| パラメータ | 型 | デフォルト | 説明 |
|-----------|------|-----------|------|
| instruction | string | 必須 | コード補完の指示 |
| input | string | "" | コードコンテキスト |
| context | string | "" | LSP情報（重要） |
| max_tokens | number | 2048 | 最大生成トークン数 |
| min_tokens | number | 10 | 最小生成トークン数 |
| temperature | number | 0.1 | サンプリング温度 |
| validate | boolean | context依存 | 出力検証有効化 |
| mode | string | "auto" | 生成モード |
| auto_retry | boolean | true | 低confidence時の自動リトライ |

### 1.2 complete_code ツール

シンプルなコード補完。prefix/suffixを指定してFill-in-the-Middle補完を行います。

```typescript
mcp__mochi__complete_code({
  prefix: "const users = await db.",
  suffix: ";\nconsole.log(users);",
  language: "typescript"
})
```

### 1.3 suggest_pattern ツール

コードパターンの提案。設計時のアイデア出しに有用です。

```typescript
mcp__mochi__suggest_pattern({
  goal: "DuckDB query handler",
  context: "// Existing imports..."
})
```

### 1.4 generate_diff ツール

差分形式でのコード変更提案。

```typescript
mcp__mochi__generate_diff({
  original_code: "function add(a, b) { return a + b; }",
  change_description: "Add type annotations",
  language: "typescript",
  context: "// TypeScript types..."
})
```

---

## 2. 高品質出力を得るパターン

### 2.1 context パラメータの活用（最重要）

**これが最も重要な改善ポイントです。**

contextにLSP情報を渡すことで:
- confidence 1.0 を達成可能
- hallucination_rate 0.0 を達成可能
- 正確なAPI名を使用

```typescript
// 高品質出力パターン
mcp__mochi__domain_query({
  instruction: "Fill in the typescript code",
  input: "// File: src/db/client.ts\nconst users = await db.",
  context: `// Available methods on DuckDBClient:
//   all<T>(sql: string, params?: any[]): Promise<T[]>
//   run(sql: string, params?: any[]): Promise<void>
//   prepare(sql: string): Statement`,
  validate: true,
  mode: "auto"
})

// 結果
{
  "response": "all<User>('SELECT * FROM users WHERE active = ?', [true])",
  "confidence": 1.0,
  "validation": {
    "is_valid": true,
    "hallucination_rate": 0.0
  },
  "mode_used": "auto",
  "retried": false
}
```

### 2.2 mode パラメータの使い分け

| モード | temperature | 用途 |
|--------|-------------|------|
| `auto` | 0.1 → 0.05 | デフォルト。低confidence時に自動リトライ |
| `conservative` | 0.05 | 高精度が必要な場合 |
| `creative` | 0.3 | 多様な提案が欲しい場合 |

```typescript
// 高精度モード
mcp__mochi__domain_query({
  instruction: "...",
  input: "...",
  mode: "conservative"
})

// 多様性重視モード
mcp__mochi__domain_query({
  instruction: "...",
  input: "...",
  mode: "creative"
})
```

### 2.3 validate オプション

contextを渡した場合、`validate: true` で出力を検証できます。

```typescript
{
  "validation": {
    "is_valid": true,           // 全APIが有効
    "hallucination_rate": 0.0,  // 幻覚率
    "hallucinated_methods": [], // 検出された幻覚
    "suggestions": {}           // 修正提案
  }
}
```

---

## 3. KIRIとの連携ワークフロー

mochiを最大限活用するには、KIRIとの連携が効果的です。

### Step 1: KIRIでコード検索

```typescript
// KIRIで関連コードを検索
mcp__kiri__context_bundle({
  query: "DuckDB query handling",
  k: 5
})
```

### Step 2: LSP情報をcontextに変換

検索結果から、使用するクラス/メソッドの情報を抽出:

```typescript
const context = `// Available methods on DuckDBClient:
//   all<T>(sql: string, params?: any[]): Promise<T[]>
//   run(sql: string, params?: any[]): Promise<void>

// Types:
//   interface User { id: string; name: string; active: boolean; }`;
```

### Step 3: mochiで生成（validate: true）

```typescript
mcp__mochi__domain_query({
  instruction: "Fill in the typescript code",
  input: "// File: src/users/repository.ts\nasync function getActiveUsers(): Promise<User[]> {\n  return await db.",
  context: context,
  validate: true
})
```

### 自動連携（将来構想）

```typescript
// autoContextオプション（未実装）
mcp__mochi__domain_query({
  instruction: "...",
  input: "...",
  autoContext: true  // KIRIから自動取得
})
```

---

## 4. トラブルシューティング

### 4.1 低confidence時の対処

**症状:** confidence < 0.5

**原因:**
- コンテキスト不足
- 曖昧な指示
- 学習データにないパターン

**対処:**
1. `context` パラメータにLSP情報を追加
2. `input` により具体的なコードを含める
3. `mode: "conservative"` を試す

### 4.2 ハルシネーション発生時

**症状:** validation.is_valid = false

**原因:**
- contextに含まれないAPIを使用

**対処:**
1. contextにより多くのメソッド情報を追加
2. validation.suggestionsの修正案を確認
3. 生成結果を手動で修正

### 4.3 出力が短すぎる

**症状:** 期待より短い出力

**対処:**
1. `min_tokens` を増やす（デフォルト: 10）
2. `max_tokens` を確認（デフォルト: 2048）
3. inputにより多くのコンテキストを含める

### 4.4 繰り返し出力

**症状:** 同じパターンの繰り返し

**対処:**
1. `mode: "creative"` を試す（repetition_penalty が低い）
2. instructionを具体的にする

---

## 5. パラメータリファレンス

### domain_query

| パラメータ | 型 | デフォルト | 説明 |
|-----------|------|-----------|------|
| instruction | string | 必須 | コード補完指示。例: "Fill in the typescript code" |
| input | string | "" | コードコンテキスト。ファイルパスコメント推奨 |
| context | string | "" | LSP情報。メソッド一覧、型定義など |
| max_tokens | number | 2048 | 最大生成トークン数 |
| min_tokens | number | 10 | 最小生成トークン数 |
| temperature | number | 0.1 | サンプリング温度 (0.0-1.0) |
| validate | boolean | context依存 | 出力検証。contextがあれば自動有効 |
| mode | string | "auto" | "auto", "conservative", "creative" |
| auto_retry | boolean | true | 低confidence時の自動リトライ |

### レスポンス

```typescript
{
  "response": string,           // 生成されたコード
  "confidence": number,         // 信頼度 (0.0-1.0)
  "inference_time_ms": number,  // 推論時間
  "tokens_generated": number,   // 生成トークン数
  "mode_used": string,          // 使用されたモード
  "retried": boolean,           // リトライされたか
  "warning": string | undefined,          // 警告メッセージ
  "alternative_action": string | undefined, // 推奨アクション
  "validation": {               // validate: true の場合
    "is_valid": boolean,
    "hallucination_rate": number,
    "hallucinated_methods": string[],
    "suggestions": object
  }
}
```

---

## 6. 推奨instruction例

### コード補完

```
Fill in the typescript code
Implement the following typescript code based on the context
Write the implementation for this typescript function
Complete the code
```

### 特定タスク

```
Add error handling to this function
Implement the missing method
Complete the database query
Add type annotations
```

### YAML補完

```
Complete the following YAML configuration
Fill in the YAML config values
Continue this YAML configuration file
```

---

## 7. confidence解釈ガイド

| confidence | 解釈 | 推奨アクション |
|------------|------|---------------|
| 0.8 - 1.0 | 高信頼 | そのまま使用可能 |
| 0.5 - 0.8 | 中信頼 | レビュー推奨 |
| 0.3 - 0.5 | 低信頼 | 検証必須。context追加を検討 |
| 0.0 - 0.3 | 非常に低い | 再生成推奨。入力を見直す |

---

## 8. まとめ

mochiを効果的に使うための3つのポイント:

1. **contextを活用する** - LSP情報を渡すことで精度が劇的に向上
2. **validate: trueを使う** - ハルシネーションを検出・防止
3. **KIRIと連携する** - コード検索 → LSP情報抽出 → mochi生成

```typescript
// 理想的な使用パターン
mcp__mochi__domain_query({
  instruction: "Fill in the typescript code",
  input: "// File: src/...\n具体的なコード文脈",
  context: "// Available methods on ClassName:\n//   method1()\n//   method2()",
  validate: true,
  mode: "auto"
})
// → confidence 1.0, hallucination_rate 0.0
```
