# Mochi MCP Server Design (ELD Phase 2)

## Issue Contract

### 目的
Fine-tuned SLM (Mochi) をMCPツール/リソースとして公開し、Claude Codeと連携させる

### 不変条件
- MCPプロトコル（JSON-RPC 2.0）準拠
- 推論レスポンス < 5秒
- kiriとの親和性（補完的な役割）

### 物差し（Acceptance Criteria）
1. MCPツール経由でドメイン固有の質問に回答できる
2. Claude Code設定でMochiサーバーに接続できる
3. リソースURIでパターン/規約を取得できる

### 停止条件
- SLM推論が不安定（一貫性のない出力）
- Apple Siliconでメモリ不足

---

## Term Catalog

### T-adapter: LoRAアダプター
- **定義**: Fine-tuned重み差分（PEFT形式）
- **境界**: `adapter_config.json` + `adapter_model.safetensors`
- **観測写像**: ファイル存在チェック + PeftModel.from_pretrained成功

### T-inference: 推論
- **定義**: SLMによるテキスト生成
- **境界**: 入力プロンプト → 出力テキスト
- **観測写像**: generate()呼び出し結果

### T-domain-query: ドメインクエリ
- **定義**: ドメイン固有の質問（Alpaca形式）
- **境界**: instruction + input → response
- **観測写像**: JSON入力のバリデーション

### T-pattern: コードパターン
- **定義**: 学習済みプロジェクトから抽出したパターン
- **境界**: パターン名 + 説明 + コード例
- **観測写像**: パターンカタログへの登録

### T-convention: 規約
- **定義**: プロジェクトのコーディング規約
- **境界**: 規約名 + 説明 + 例
- **観測写像**: 規約カタログへの登録

---

## Law Catalog

### L-response-time: レスポンス時間制約
- **種別**: Performance Constraint
- **条件**: 推論レスポンス < 5秒
- **スコープ**: S1（アプリケーション全体）
- **違反時動作**: タイムアウトエラー（504）を返却
- **検証手段**: 推論時間計測 + しきい値チェック
- **接地**: ユニットテスト（タイムアウト検証）

### L-mcp-compliance: MCPプロトコル準拠
- **種別**: Interface Contract
- **条件**: JSON-RPC 2.0形式で通信
- **スコープ**: S0（外部インターフェース）
- **違反時動作**: バリデーションエラー返却
- **検証手段**: Zodスキーマバリデーション
- **接地**: 統合テスト（MCP Inspector）

### L-adapter-required: アダプター必須
- **種別**: Precondition
- **条件**: 推論前にアダプターがロードされている
- **スコープ**: S1（サーバー起動時）
- **違反時動作**: 503 Service Unavailable
- **検証手段**: self.model is not None チェック
- **接地**: 起動シーケンステスト

### L-memory-bound: メモリ制限
- **種別**: Resource Constraint
- **条件**: 推論時メモリ使用量 < 32GB
- **スコープ**: S1（Apple Silicon環境）
- **違反時動作**: OOMガード発動、推論拒否
- **検証手段**: psutil.virtual_memory()監視
- **接地**: 負荷テスト

### L-idempotent-resource: リソース冪等性
- **種別**: Invariant
- **条件**: 同じリソースURIは同じ内容を返す
- **スコープ**: S0（リソースAPI）
- **違反時動作**: なし（設計で保証）
- **検証手段**: 同一URI複数回呼び出し
- **接地**: ユニットテスト

---

## MCP Tools設計

### 1. domain_query
ドメイン固有の質問に回答する

```typescript
// Tool Definition
{
  name: "domain_query",
  description: "Ask domain-specific questions about the learned codebase",
  inputSchema: {
    type: "object",
    properties: {
      instruction: {
        type: "string",
        description: "The question or instruction"
      },
      input: {
        type: "string",
        description: "Additional context (optional)"
      },
      max_tokens: {
        type: "number",
        description: "Maximum tokens to generate (default: 512)"
      }
    },
    required: ["instruction"]
  }
}

// Output Schema
{
  response: string,       // SLM生成テキスト
  confidence: number,     // 推論信頼度 (0-1)
  inference_time_ms: number,
  tokens_generated: number
}
```

### 2. complete_code
コード補完を行う

```typescript
{
  name: "complete_code",
  description: "Complete code based on domain patterns",
  inputSchema: {
    type: "object",
    properties: {
      prefix: {
        type: "string",
        description: "Code before cursor"
      },
      suffix: {
        type: "string",
        description: "Code after cursor (optional)"
      },
      language: {
        type: "string",
        description: "Programming language"
      }
    },
    required: ["prefix"]
  }
}

// Output Schema
{
  completion: string,
  alternatives: string[],  // 最大3つの候補
  inference_time_ms: number
}
```

### 3. suggest_pattern
適切なパターンを提案する

```typescript
{
  name: "suggest_pattern",
  description: "Suggest relevant code patterns from learned knowledge",
  inputSchema: {
    type: "object",
    properties: {
      context: {
        type: "string",
        description: "Current code context"
      },
      goal: {
        type: "string",
        description: "What you're trying to achieve"
      }
    },
    required: ["goal"]
  }
}

// Output Schema
{
  patterns: Array<{
    name: string,
    description: string,
    example: string,
    relevance: number
  }>,
  inference_time_ms: number
}
```

---

## MCP Resources設計

### Resource URI Pattern
```
mochi://patterns/{pattern_name}
mochi://conventions/{convention_name}
mochi://examples/{example_id}
mochi://stats
```

### 1. patterns/* - コードパターン
```typescript
// URI: mochi://patterns/error-handling
{
  uri: "mochi://patterns/error-handling",
  name: "Error Handling Pattern",
  mimeType: "text/markdown",
  description: "Common error handling patterns in this project"
}
```

### 2. conventions/* - 規約
```typescript
// URI: mochi://conventions/naming
{
  uri: "mochi://conventions/naming",
  name: "Naming Conventions",
  mimeType: "text/markdown",
  description: "Variable and function naming conventions"
}
```

### 3. stats - 統計情報
```typescript
// URI: mochi://stats
{
  uri: "mochi://stats",
  name: "Model Statistics",
  mimeType: "application/json",
  description: "Training statistics and model info"
}
```

---

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                       Claude Code                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ MCP (JSON-RPC 2.0)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Mochi MCP Server                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Tool Handler │  │   Resource   │  │   Runtime    │      │
│  │              │  │   Handler    │  │              │      │
│  │ domain_query │  │ patterns/*   │  │ Model Loader │      │
│  │ complete_code│  │ conventions/*│  │ Memory Guard │      │
│  │ suggest_ptn  │  │ stats        │  │ Timeout Ctrl │      │
│  └──────┬───────┘  └──────────────┘  └──────┬───────┘      │
│         │                                     │              │
│         └─────────────┬───────────────────────┘              │
│                       ▼                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Inference Engine                        │    │
│  │  ┌─────────────┐    ┌─────────────┐                 │    │
│  │  │ Base Model  │ +  │ LoRA Adapter │                │    │
│  │  │ Qwen2.5-1.5B│    │ (domain)     │                │    │
│  │  └─────────────┘    └─────────────┘                 │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ (kiriと連携時)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      kiri MCP Server                         │
│  context_bundle, files_search, snippets_get ...             │
└─────────────────────────────────────────────────────────────┘
```

---

## 実装計画

### Phase 1: Core Server (MVP)
1. MCPサーバー骨格（stdio/http両対応）
2. domain_queryツール実装
3. 推論エンジン統合
4. L-response-time, L-adapter-required実装

### Phase 2: Extended Tools
1. complete_codeツール
2. suggest_patternツール
3. パターン/規約抽出機能

### Phase 3: Resources & Integration
1. リソースハンドラー実装
2. パターンカタログ構築
3. kiri連携テスト

---

## Evidence Ladder計画

| Level | 内容 | 対象Law |
|-------|------|---------|
| L0 | 型チェック（mypy） | 全Law |
| L1 | ユニットテスト | L-adapter-required, L-idempotent-resource |
| L2 | 統合テスト（MCP Inspector） | L-mcp-compliance |
| L2 | パフォーマンステスト | L-response-time |
| L3 | メモリ負荷テスト | L-memory-bound |

---

## Link Map

```
L-response-time ─────► T-inference
L-mcp-compliance ───► T-domain-query
L-adapter-required ──► T-adapter
L-memory-bound ─────► T-inference
L-idempotent-resource ► T-pattern, T-convention
```

孤立チェック: すべてのTerm/Lawが相互参照されていることを確認済み。
