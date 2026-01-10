# データパイプライン設計

## 概要

プロジェクトの資産（コード、ドキュメント、履歴）を学習可能なデータセットに変換するパイプライン。

```
[Raw Sources] → [Ingestion] → [Preprocessing] → [Data Generation] → [Dataset]
```

## 1. データソース

### 1.1 ソースコード

| 対象 | 抽出内容 | 用途 |
|------|----------|------|
| ソースファイル | コード全体 | コード補完、生成 |
| 関数/メソッド | シグネチャ+本体 | 関数生成 |
| クラス定義 | 構造+メソッド | クラス設計理解 |
| 型定義 | インターフェース、型 | 型安全な生成 |
| インポート文 | 依存関係 | モジュール理解 |

### 1.2 ドキュメント

| 対象 | 抽出内容 | 用途 |
|------|----------|------|
| README | プロジェクト概要 | 全体理解 |
| API仕様書 | エンドポイント定義 | API生成 |
| 設計書 | アーキテクチャ | 設計意図理解 |
| コメント/JSDoc | インラインドキュメント | コード意図理解 |

### 1.3 履歴情報

| 対象 | 抽出内容 | 用途 |
|------|----------|------|
| コミット | メッセージ+差分 | 変更パターン学習 |
| Issue | 問題記述+解決 | 問題解決パターン |
| PR | レビューコメント | コード品質基準 |

## 2. 前処理パイプライン

### 2.1 コードチャンク化

```python
from dataclasses import dataclass
from enum import Enum

class ChunkGranularity(Enum):
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    BLOCK = "block"

@dataclass
class CodeChunk:
    content: str
    file_path: str
    start_line: int
    end_line: int
    granularity: ChunkGranularity
    language: str
    metadata: dict  # imports, dependencies, etc.
```

**チャンク化戦略**

```
┌─────────────────────────────────────┐
│            ソースファイル            │
├─────────────────────────────────────┤
│  import statements                  │ ← 依存情報として抽出
├─────────────────────────────────────┤
│  class UserService:                 │ ← クラス単位チャンク
│    def __init__(self):              │   ├─ メソッド単位チャンク
│      ...                            │   │
│    def create_user(self, data):     │   ├─ メソッド単位チャンク
│      ...                            │   │
│    def delete_user(self, id):       │   └─ メソッド単位チャンク
│      ...                            │
├─────────────────────────────────────┤
│  def helper_function():             │ ← 関数単位チャンク
│    ...                              │
└─────────────────────────────────────┘
```

### 2.2 AST解析

```python
import ast
from tree_sitter import Language, Parser

class CodeAnalyzer:
    """多言語対応のコード解析"""

    SUPPORTED_LANGUAGES = [
        "python", "javascript", "typescript",
        "go", "rust", "java", "swift"
    ]

    def extract_functions(self, code: str, lang: str) -> list[FunctionInfo]:
        """関数/メソッドの抽出"""
        ...

    def extract_classes(self, code: str, lang: str) -> list[ClassInfo]:
        """クラス定義の抽出"""
        ...

    def extract_imports(self, code: str, lang: str) -> list[ImportInfo]:
        """インポート文の抽出"""
        ...
```

### 2.3 重複排除

```python
from datasketch import MinHash, MinHashLSH

class Deduplicator:
    """類似コードの重複排除"""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.lsh = MinHashLSH(threshold=threshold, num_perm=128)

    def deduplicate(self, chunks: list[CodeChunk]) -> list[CodeChunk]:
        """MinHashLSHによる重複排除"""
        ...
```

## 3. 学習データ生成

### 3.1 コード補完ペア

```python
@dataclass
class CompletionPair:
    prefix: str      # 補完前のコード
    completion: str  # 補完内容
    context: str     # 周辺コンテキスト

def generate_completion_pairs(chunk: CodeChunk) -> list[CompletionPair]:
    """
    コードチャンクから補完ペアを生成

    戦略:
    1. 関数本体の途中で切断
    2. 引数リストの途中で切断
    3. 文の途中で切断
    """
    pairs = []

    # 例: 関数の途中で切断
    # prefix: "def calculate_total(items):\n    total = 0\n    for item in"
    # completion: " items:\n        total += item.price\n    return total"

    return pairs
```

### 3.2 Instruction-Response ペア

```python
@dataclass
class InstructionPair:
    instruction: str  # 指示文
    context: str      # コンテキスト（既存コード）
    response: str     # 期待される出力

# 生成例
examples = [
    InstructionPair(
        instruction="UserServiceクラスにユーザー削除メソッドを追加してください",
        context="class UserService:\n    def create_user(self, data): ...",
        response="def delete_user(self, user_id: int) -> bool:\n    ..."
    ),
    InstructionPair(
        instruction="この関数にエラーハンドリングを追加してください",
        context="def fetch_data(url):\n    response = requests.get(url)\n    return response.json()",
        response="def fetch_data(url):\n    try:\n        response = requests.get(url)\n        response.raise_for_status()\n        return response.json()\n    except RequestException as e:\n        logger.error(f'Failed to fetch: {e}')\n        raise"
    )
]
```

### 3.3 LLMによる合成データ生成

```python
class SyntheticDataGenerator:
    """LLMを活用した学習データの拡張"""

    def __init__(self, llm_client):
        self.llm = llm_client

    async def generate_qa_pairs(self, code: str) -> list[dict]:
        """コードに関するQ&Aペアを生成"""
        prompt = f"""
        以下のコードについて、開発者が質問しそうな内容と
        その回答のペアを5つ生成してください。

        JSON形式で出力:
        [{{"question": "...", "answer": "..."}}]

        コード:
        ```
        {code}
        ```
        """
        return await self.llm.generate(prompt)

    async def generate_variations(self, code: str) -> list[str]:
        """コードのバリエーションを生成（データ拡張）"""
        ...

    async def generate_explanations(self, code: str) -> str:
        """コードの説明を生成"""
        ...
```

## 4. データセットフォーマット

### 4.1 Alpaca形式

```json
{
  "instruction": "ユーザー認証を行う関数を実装してください",
  "input": "既存のUserRepositoryクラスを使用してください",
  "output": "async def authenticate(username: str, password: str) -> User | None:\n    user = await user_repo.find_by_username(username)\n    if user and verify_password(password, user.hashed_password):\n        return user\n    return None"
}
```

### 4.2 ShareGPT形式

```json
{
  "conversations": [
    {"from": "human", "value": "UserServiceのcreate_userメソッドを見せて"},
    {"from": "gpt", "value": "```python\nasync def create_user(self, data: CreateUserDto) -> User:\n    ...\n```"},
    {"from": "human", "value": "バリデーションを追加して"},
    {"from": "gpt", "value": "```python\nasync def create_user(self, data: CreateUserDto) -> User:\n    if not data.email:\n        raise ValidationError('Email is required')\n    ...\n```"}
  ]
}
```

## 5. パイプライン実行

```python
from mochi.pipeline import Pipeline

# パイプライン定義
pipeline = Pipeline([
    GitIngestion(repo_path="./target-project"),
    DocIngestion(doc_paths=["./docs", "./README.md"]),
    CodeChunker(granularity="function"),
    Deduplicator(threshold=0.8),
    CompletionPairGenerator(),
    InstructionPairGenerator(),
    SyntheticDataGenerator(llm=claude_client),
    DatasetExporter(format="alpaca", output_path="./dataset")
])

# 実行
dataset = await pipeline.run()
print(f"Generated {len(dataset)} training examples")
```

## 6. 品質管理

### フィルタリング基準

- 最小トークン数: 10
- 最大トークン数: 2048
- 重複スコア閾値: 0.8
- 言語検出信頼度: 0.9以上

### 検証チェック

- [ ] 構文エラーのあるコードを除外
- [ ] 機密情報（APIキー等）のスキャン・除外
- [ ] ライセンス互換性の確認
- [ ] 文字エンコーディングの正規化

## 7. 自然言語処理（多言語対応）

### 課題

プロジェクトのコメントやドキュメントには日本語などの非英語テキストが含まれることがある。

| 項目 | 課題 |
|------|------|
| **トークン効率** | 日本語は1文字→2-4トークンに分割され、コンテキスト効率が悪化 |
| **ベースモデル** | コード特化モデルは英語中心のものが多い |
| **学習品質** | 言語混在がモデルの混乱を招く可能性 |

### トークン効率の比較

```
# 同じ意味のコメント

英語:   "Get the user's name"     → ~5 tokens
日本語: "ユーザー名を取得する"      → ~12 tokens (約2.4倍)

# モデル別の日本語トークン効率
Qwen2.5:    比較的効率的（多言語設計）
CodeLlama:  非効率（英語中心設計、3-4倍のトークン消費）
StarCoder:  非効率（欧州言語中心）
```

### 処理戦略

```
┌─────────────────────────────────────────────────────────────┐
│                    言語処理フロー                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [入力]                                                      │
│    ├─ ソースコード（コメント含む）                           │
│    └─ ドキュメント（日本語Markdown等）                       │
│           │                                                  │
│           ▼                                                  │
│  [言語検出 & 分類]                                           │
│    ├─ コード部分 → そのまま保持                              │
│    └─ 自然言語部分 → 処理方針を選択                          │
│           │                                                  │
│           ▼                                                  │
│  [処理オプション] ← ユーザー設定                             │
│    ├─ preserve:  原文保持（推奨）                            │
│    ├─ translate: 英語に翻訳                                  │
│    └─ remove:    コメント/docstring除去                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 設定オプション

```yaml
# mochi.yaml

# 推奨設定（日本語プロジェクト向け）
language_handling:
  strategy: preserve              # preserve | translate | remove
  base_model: Qwen/Qwen2.5-Coder-7B  # 多言語対応モデル推奨

  # 対象別の設定
  comments: preserve              # インラインコメント
  docstrings: preserve            # docstring/JSDoc
  markdown: preserve              # ドキュメント

  # 翻訳を使う場合
  translation:
    enabled: false
    api: claude                   # claude | openai | deepl
    source_languages: [ja, zh, ko]
    target_language: en
    cache: true                   # 翻訳結果をキャッシュ
```

### 実装

```python
from langdetect import detect
from dataclasses import dataclass
from enum import Enum

class LanguageStrategy(Enum):
    PRESERVE = "preserve"
    TRANSLATE = "translate"
    REMOVE = "remove"

@dataclass
class LanguageConfig:
    strategy: LanguageStrategy
    comments: LanguageStrategy
    docstrings: LanguageStrategy
    markdown: LanguageStrategy
    translation_api: str | None = None

class LanguageHandler:
    """自然言語テキストの処理"""

    def __init__(self, config: LanguageConfig, translator=None):
        self.config = config
        self.translator = translator

    def process(self, text: str, text_type: str = "comment") -> str:
        """
        テキストを設定に基づいて処理

        Args:
            text: 処理対象テキスト
            text_type: "comment" | "docstring" | "markdown"
        """
        strategy = getattr(self.config, text_type + "s", self.config.strategy)

        if strategy == LanguageStrategy.PRESERVE:
            return text

        if strategy == LanguageStrategy.REMOVE:
            return ""

        if strategy == LanguageStrategy.TRANSLATE:
            detected_lang = self._detect_language(text)
            if detected_lang != "en":
                return self._translate(text, target="en")
            return text

        return text

    def _detect_language(self, text: str) -> str:
        """言語を検出"""
        try:
            return detect(text)
        except:
            return "unknown"

    def _translate(self, text: str, target: str) -> str:
        """翻訳を実行"""
        if self.translator is None:
            raise ValueError("Translator not configured")
        return self.translator.translate(text, target=target)

class CommentExtractor:
    """コードからコメントを抽出・処理"""

    def __init__(self, lang_handler: LanguageHandler):
        self.lang_handler = lang_handler

    def process_code(self, code: str, language: str) -> tuple[str, list[str]]:
        """
        コードを処理し、コメントを設定に従って変換

        Returns:
            (processed_code, extracted_comments)
        """
        comments = self._extract_comments(code, language)
        processed_comments = [
            self.lang_handler.process(c, "comment") for c in comments
        ]
        # コメントを置換したコードを返す
        return self._replace_comments(code, comments, processed_comments)
```

### ベースモデル別の日本語対応

| モデル | 日本語対応 | トークン効率 | 推奨度 |
|--------|----------|-------------|-------|
| **Qwen2.5-Coder** | 良好 | 高 | 日本語プロジェクトに最適 |
| Phi-3 | 中程度 | 中 | 英語中心なら可 |
| CodeLlama | 弱い | 低 | 非推奨 |
| StarCoder2 | 弱い | 低 | 非推奨 |
| DeepSeek-Coder | 中程度 | 中 | 代替候補 |

### 推奨事項

1. **日本語を多用するプロジェクト**
   - ベースモデル: `Qwen2.5-Coder` を強く推奨
   - 言語処理: `preserve`（原文保持）
   - 理由: 日本語コメント/ドキュメントはプロジェクト固有の文脈を含む重要な情報

2. **英語統一が必要な場合**
   - 言語処理: `translate`
   - 翻訳API: Claude または DeepL（コード文脈の理解が必要）
   - 注意: 翻訳コストと品質劣化のトレードオフ

3. **コードのみ重視する場合**
   - 言語処理: `remove`
   - 用途: トークン効率を最大化したい場合
   - 注意: 文脈情報の損失あり
