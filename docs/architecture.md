# アーキテクチャ設計

## システム全体構成

```
┌────────────────────────────────────────────────────────────────┐
│                        Data Ingestion Layer                    │
├────────────────────────────────────────────────────────────────┤
│  Git Connector │ Doc Parser │ Issue Tracker │ CI/CD Logs      │
└───────┬────────┴─────┬──────┴──────┬────────┴───────┬─────────┘
        │              │             │                │
        ▼              ▼             ▼                ▼
┌────────────────────────────────────────────────────────────────┐
│                    Preprocessing Pipeline                       │
├────────────────────────────────────────────────────────────────┤
│  Code Chunker │ Doc Normalizer │ Context Builder │ Deduplicator│
└───────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│                    Training Data Generator                      │
├────────────────────────────────────────────────────────────────┤
│  - コード補完ペア生成                                           │
│  - ドキュメント⇔コード対応付け                                  │
│  - Q&Aデータセット自動生成（LLM活用）                           │
│  - コミットメッセージ⇔差分ペア                                  │
└───────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│                      Training Engine                            │
├────────────────────────────────────────────────────────────────┤
│  Base Model: Phi-3-mini / Gemma-2B / CodeGemma / Qwen-1.5      │
│  Method: LoRA / QLoRA / Full Fine-tuning                       │
│  Framework: Hugging Face Transformers + PEFT                   │
└───────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│                    Deployment & Integration                     │
├────────────────────────────────────────────────────────────────┤
│  GGUF Export │ Ollama │ vLLM │ IDE Plugin │ Claude Code連携   │
└────────────────────────────────────────────────────────────────┘
```

## コンポーネント詳細

### 1. Data Ingestion Layer

データソースからの情報収集を担当。

```
mochi/
├── ingestion/
│   ├── git_connector.py      # Gitリポジトリからのコード取得
│   ├── doc_parser.py         # Markdown/RST/AsciiDocパーサー
│   ├── issue_tracker.py      # GitHub/GitLab/Jira連携
│   └── ci_logs.py            # CI/CDログ収集（オプション）
```

**Git Connector**
- クローン/プル操作
- ブランチ・タグ管理
- コミット履歴の取得
- ファイル変更差分の抽出

**Doc Parser**
- Markdownパース
- コードブロック抽出
- セクション構造化

### 2. Preprocessing Pipeline

生データを学習可能な形式に変換。

```
mochi/
├── preprocessing/
│   ├── code_chunker.py       # コードの意味単位分割
│   ├── doc_normalizer.py     # ドキュメント正規化
│   ├── context_builder.py    # コンテキスト情報付与
│   └── deduplicator.py       # 重複排除
```

**Code Chunker**
```python
# チャンク化の戦略
class ChunkStrategy:
    FUNCTION = "function"      # 関数単位
    CLASS = "class"            # クラス単位
    FILE = "file"              # ファイル単位
    SLIDING_WINDOW = "sliding" # スライディングウィンドウ
```

### 3. Training Data Generator

学習データセットの自動生成。

```
mochi/
├── data_generation/
│   ├── completion_pairs.py   # コード補完ペア生成
│   ├── doc_code_mapping.py   # ドキュメント⇔コード対応
│   ├── qa_generator.py       # Q&A生成（LLM活用）
│   └── commit_pairs.py       # コミット⇔差分ペア
```

**データフォーマット**
```json
{
  "instruction": "UserServiceクラスのcreateUserメソッドを実装してください",
  "context": "// 既存のUserServiceクラス定義...",
  "response": "async createUser(data: CreateUserDto): Promise<User> { ... }"
}
```

### 4. Training Engine

モデルのファインチューニングを実行。

```
mochi/
├── training/
│   ├── trainer.py            # 学習ループ
│   ├── config.py             # 学習設定
│   ├── lora_config.py        # LoRA/QLoRA設定
│   └── evaluation.py         # 評価メトリクス
```

### 5. Deployment

学習済みモデルのエクスポートと配信。

```
mochi/
├── deployment/
│   ├── exporter.py           # GGUF/SafeTensors変換
│   ├── ollama_integration.py # Ollama Modelfile生成
│   └── server.py             # 推論サーバー
```

## SLMの利用パターン

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   開発者     │─────▶│ ドメインSLM │─────▶│    LLM      │
│  (クエリ)   │      │ (コンテキスト│      │ (最終生成)  │
└─────────────┘      │  強化・提案) │      └─────────────┘
                     └─────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         コード規約    型情報補完   関連コード
         の提示        の提案       の参照
```

### 統合パターン

1. **プロンプト強化**: SLMがコンテキスト情報を生成 → LLMに渡す
2. **リランキング**: LLMの複数候補をSLMがスコアリング
3. **検証**: LLMの出力をSLMがプロジェクト規約に照らして検証

## ディレクトリ構成（予定）

```
mochi/
├── src/
│   └── mochi/
│       ├── __init__.py
│       ├── cli.py                # CLIエントリーポイント
│       ├── ingestion/            # データ取り込み
│       ├── preprocessing/        # 前処理
│       ├── data_generation/      # 学習データ生成
│       ├── training/             # 学習エンジン
│       └── deployment/           # デプロイ
├── tests/
├── docs/
├── pyproject.toml
└── README.md
```
