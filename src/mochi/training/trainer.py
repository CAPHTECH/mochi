"""Training module for QLoRA fine-tuning."""

from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


@dataclass
class QLoRAConfig:
    """Configuration for QLoRA training."""

    # Model
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B"
    max_seq_length: int = 2048

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Training
    num_epochs: int = 2
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01

    # Output
    output_dir: str = "./output"
    logging_steps: int = 10
    save_steps: int = 100


def format_alpaca_prompt(example: dict) -> str:
    """Format an Alpaca example into a prompt string."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    else:
        prompt = f"""### Instruction:
{instruction}

### Response:
{output}"""

    return prompt


class MochiTrainer:
    """Trainer for domain-specific SLM fine-tuning."""

    def __init__(self, config: QLoRAConfig | None = None) -> None:
        """Initialize trainer with configuration."""
        self.config = config or QLoRAConfig()
        self.model = None
        self.tokenizer = None

    def setup(self) -> None:
        """Set up model and tokenizer with QLoRA configuration."""
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        # LoRA config
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def train(
        self,
        train_file: str | Path,
        eval_file: str | Path | None = None,
    ) -> None:
        """
        Run training on the dataset.

        Args:
            train_file: Path to training JSONL file
            eval_file: Path to evaluation JSONL file (optional)
        """
        if self.model is None or self.tokenizer is None:
            self.setup()

        # Load dataset
        data_files = {"train": str(train_file)}
        if eval_file:
            data_files["eval"] = str(eval_file)

        dataset = load_dataset("json", data_files=data_files)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            optim="adamw_8bit",
            fp16=False,
            bf16=True,
            logging_steps=self.config.logging_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            eval_strategy="steps" if eval_file else "no",
            eval_steps=self.config.save_steps if eval_file else None,
            max_grad_norm=0.3,
            group_by_length=True,
            report_to="none",
        )

        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("eval"),
            args=training_args,
            formatting_func=lambda x: format_alpaca_prompt(x),
            max_seq_length=self.config.max_seq_length,
            packing=True,
        )

        # Train
        trainer.train()

        # Save the adapter
        adapter_path = Path(self.config.output_dir) / "adapter"
        trainer.save_model(str(adapter_path))

        return adapter_path

    def merge_and_save(self, adapter_path: str | Path, output_path: str | Path) -> None:
        """Merge LoRA adapter with base model and save."""
        from peft import PeftModel

        # Load base model (without quantization for merging)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load and merge adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
        merged_model = model.merge_and_unload()

        # Save
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
