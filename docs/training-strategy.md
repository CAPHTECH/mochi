# 学習戦略

## 1. ベースモデル選定

### 候補モデル比較

| モデル | パラメータ数 | 特徴 | 日本語対応 | 推奨用途 |
|--------|-------------|------|-----------|----------|
| **Qwen2.5-Coder** | 1.5B / 7B | コード特化、多言語対応 | 良好 | 主力候補（日本語プロジェクトに最適） |
| **Phi-3-mini** | 3.8B | 高効率、推論性能良好 | 中程度 | バランス重視（英語中心プロジェクト） |
| **CodeGemma** | 2B / 7B | Google製、コード特化 | 中程度 | 代替候補 |
| **StarCoder2** | 3B / 7B | BigCode製、OSS特化 | 弱い | OSS学習向け（英語のみ推奨） |
| **DeepSeek-Coder** | 1.3B / 6.7B | コード生成特化 | 中程度 | 軽量優先 |

### 日本語対応について

日本語のコメントやドキュメントを含むプロジェクトでは、ベースモデルの多言語対応が重要。

**Qwen2.5-Coderを推奨する理由:**
- Alibabaが開発、多言語前提のトークナイザー設計
- 日本語テキストのトークン効率が他モデルより優れている
- 日本語⇔コードの文脈理解が比較的良好

詳細は [データパイプライン - 自然言語処理](./data-pipeline.md#7-自然言語処理多言語対応) を参照。

### 選定基準

```
スコア = (コード性能 × 0.4) + (推論速度 × 0.3) + (メモリ効率 × 0.2) + (ライセンス × 0.1)
```

**推奨構成**
- **開発・テスト**: Qwen2.5-Coder-1.5B（軽量、高速イテレーション）
- **本番**: Qwen2.5-Coder-7B または Phi-3-mini（品質重視）

## 2. ファインチューニング手法

### 2.1 QLoRA（推奨）

```python
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

# 4bit量子化設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# LoRA設定
lora_config = LoraConfig(
    r=64,                      # LoRAランク
    lora_alpha=16,             # スケーリング係数
    target_modules=[           # 適用レイヤー
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

**メリット**
- VRAM使用量: ~8GB（7Bモデル）
- 学習速度: Full fine-tuningの2-3倍
- 品質: Full fine-tuningの95%程度を維持

### 2.2 Unslothによる高速化

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-Coder-7B",
    max_seq_length=2048,
    dtype=None,  # 自動検出
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # メモリ最適化
    random_state=42,
)
```

**Unslothの利点**
- 学習速度: 2-5倍高速化
- メモリ使用量: 最大70%削減
- 追加コストなし

## 3. 学習設定

### 3.1 ハイパーパラメータ

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",

    # バッチサイズ
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # 実効バッチサイズ: 16

    # 学習率
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,

    # エポック
    num_train_epochs=3,

    # 最適化
    optim="adamw_8bit",
    weight_decay=0.01,
    max_grad_norm=0.3,

    # 精度
    fp16=False,
    bf16=True,  # A100/H100向け

    # ロギング
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,

    # 評価
    eval_strategy="steps",
    eval_steps=100,
)
```

### 3.2 データ量の目安

| プロジェクト規模 | 推定データ量 | 推奨エポック数 |
|-----------------|-------------|---------------|
| 小規模（~10kLOC） | 1k-5k examples | 5-10 |
| 中規模（~100kLOC） | 10k-50k examples | 3-5 |
| 大規模（~1MLOC） | 100k+ examples | 1-3 |

## 4. 学習パイプライン

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [Dataset]                                                   │
│      │                                                       │
│      ▼                                                       │
│  ┌─────────────┐                                            │
│  │ Tokenization │ ← max_length=2048, padding, truncation    │
│  └──────┬──────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐     ┌──────────────┐                       │
│  │ Base Model  │────▶│ LoRA Adapter │                       │
│  │  (frozen)   │     │ (trainable)  │                       │
│  └─────────────┘     └──────┬───────┘                       │
│                             │                                │
│         ┌───────────────────┼───────────────────┐           │
│         ▼                   ▼                   ▼           │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐      │
│  │   Train    │     │   Eval     │     │ Checkpoint │      │
│  │   Loss     │     │  Metrics   │     │   Save     │      │
│  └────────────┘     └────────────┘     └────────────┘      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 実装例

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=True,  # 効率的なバッチング
)

# 学習実行
trainer.train()

# アダプタ保存
trainer.save_model("./lora_adapter")
```

## 5. 評価戦略

### 5.1 自動評価メトリクス

```python
@dataclass
class EvaluationMetrics:
    # 生成品質
    perplexity: float
    bleu_score: float
    code_bleu: float

    # コード固有
    syntax_validity: float    # 構文的に正しいか
    type_correctness: float   # 型エラーがないか
    test_pass_rate: float     # 生成テストの合格率

    # プロジェクト固有
    style_compliance: float   # コーディング規約準拠率
    api_usage_accuracy: float # 内部API使用の正確性
```

### 5.2 プロジェクト固有ベンチマーク

```python
class ProjectBenchmark:
    """プロジェクト固有の評価ベンチマーク"""

    def __init__(self, project_path: str):
        self.test_cases = self._generate_test_cases(project_path)

    def _generate_test_cases(self, path: str) -> list[TestCase]:
        """
        プロジェクトから評価用テストケースを自動生成
        - 既存関数の再実装タスク
        - ドキュメントからのコード生成タスク
        - バグ修正タスク
        """
        ...

    def evaluate(self, model) -> dict:
        results = {}
        for test in self.test_cases:
            output = model.generate(test.prompt)
            results[test.id] = self._score(output, test.expected)
        return results
```

## 6. 継続学習

### 6.1 増分学習パイプライン

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  新規コミット │────▶│  差分データ  │────▶│  増分学習    │
│  検出        │     │  生成        │     │  実行        │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 ▼
                                          ┌──────────────┐
                                          │ アダプタ更新 │
                                          │ (マージ)     │
                                          └──────────────┘
```

### 6.2 アダプタ管理

```python
class AdapterManager:
    """LoRAアダプタのバージョン管理"""

    def merge_adapters(self, base: str, new: str) -> str:
        """複数アダプタのマージ"""
        ...

    def rollback(self, version: str):
        """特定バージョンへのロールバック"""
        ...

    def compare(self, v1: str, v2: str) -> dict:
        """バージョン間の性能比較"""
        ...
```

## 7. 必要リソース

### 学習環境

| 構成 | GPU | VRAM | 推定学習時間（10k examples） |
|------|-----|------|---------------------------|
| 最小 | RTX 4090 | 24GB | 2-4時間 |
| 推奨 | A100 40GB | 40GB | 1-2時間 |
| 高速 | H100 80GB | 80GB | 30分-1時間 |

### クラウドコスト目安

- **AWS**: p4d.24xlarge (A100x8) ~$32/h
- **GCP**: a2-highgpu-1g (A100x1) ~$4/h
- **Lambda Labs**: A100 (1x) ~$1.1/h
- **RunPod**: A100 (1x) ~$1.5/h
