"""Training module for LoRA fine-tuning with CUDA and Apple Silicon support."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer


class DeviceType(Enum):
    """Supported device types."""

    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


def detect_device() -> DeviceType:
    """Detect the best available device."""
    if torch.cuda.is_available():
        return DeviceType.CUDA
    elif torch.backends.mps.is_available():
        return DeviceType.MPS
    else:
        return DeviceType.CPU


# Model presets for 2025+ models
MODEL_PRESETS: dict[str, dict] = {
    "qwen3-coder": {
        "base_model": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "max_seq_length": 4096,
        "is_moe": True,
    },
    "gpt-oss": {
        "base_model": "openai/gpt-oss-20b",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "max_seq_length": 4096,
        "is_moe": True,
    },
}


@dataclass
class LoRAConfig:
    """Configuration for LoRA training (supports both CUDA and MPS)."""

    # Model
    base_model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    max_seq_length: int = 2048
    is_moe: bool = False  # Mixture of Experts model flag

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

    # Device (auto-detected if None)
    device_type: DeviceType | None = None

    # Quantization (CUDA only)
    use_4bit: bool = False  # Only works on CUDA with bitsandbytes

    @classmethod
    def from_preset(cls, preset_name: str, **overrides) -> "LoRAConfig":
        """Create config from a model preset.

        Available presets: qwen3-coder, gpt-oss
        """
        if preset_name not in MODEL_PRESETS:
            available = ", ".join(MODEL_PRESETS.keys())
            raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")

        preset = MODEL_PRESETS[preset_name]
        return cls(
            base_model=preset["base_model"],
            target_modules=preset["target_modules"],
            max_seq_length=preset["max_seq_length"],
            is_moe=preset["is_moe"],
            **overrides,
        )


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

    def __init__(self, config: LoRAConfig | None = None) -> None:
        """Initialize trainer with configuration."""
        self.config = config or LoRAConfig()
        self.model = None
        self.tokenizer = None
        self.device_type = self.config.device_type or detect_device()

    def setup(self) -> None:
        """Set up model and tokenizer with LoRA configuration."""
        print(f"Detected device: {self.device_type.value}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Device-specific model loading
        if self.device_type == DeviceType.CUDA and self.config.use_4bit:
            self._setup_cuda_quantized()
        elif self.device_type == DeviceType.MPS:
            self._setup_mps()
        elif self.device_type == DeviceType.CUDA:
            self._setup_cuda()
        else:
            self._setup_cpu()

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

    def _setup_cuda_quantized(self) -> None:
        """Set up model with 4-bit quantization (CUDA only)."""
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        print("Using CUDA with 4-bit quantization (QLoRA)")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model = prepare_model_for_kbit_training(self.model)

    def _setup_cuda(self) -> None:
        """Set up model for CUDA without quantization."""
        print("Using CUDA with fp16")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    def _setup_mps(self) -> None:
        """Set up model for Apple Silicon (MPS)."""
        print("Using Apple Silicon (MPS)")

        if self.config.is_moe:
            # MoE models (Qwen3-Coder, GPT-OSS) on MPS
            # Use float16 for MoE to fit in memory, with careful handling
            print(f"Loading MoE model: {self.config.base_model}")
            print("Note: MoE model with ~3-4B active params. Using fp16 on 128GB Mac.")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch.float16,
                device_map={"": "mps"},
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        else:
            # Standard dense models on MPS
            print("Note: MPS uses fp32 for stability. 128GB memory is sufficient for 1.5B model.")

            # MPS works best with fp32 for stability
            # For 1.5B model: ~6GB for model + gradients + optimizer states
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch.float32,
                device_map={"": "mps"},
                trust_remote_code=True,
            )

    def _setup_cpu(self) -> None:
        """Set up model for CPU."""
        print("Using CPU with fp32 (this will be slow)")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )

    def train(
        self,
        train_file: str | Path,
        eval_file: str | Path | None = None,
    ) -> Path:
        """
        Run training on the dataset.

        Args:
            train_file: Path to training JSONL file
            eval_file: Path to evaluation JSONL file (optional)

        Returns:
            Path to saved adapter
        """
        if self.model is None or self.tokenizer is None:
            self.setup()

        # Load dataset
        data_files = {"train": str(train_file)}
        if eval_file:
            data_files["eval"] = str(eval_file)

        dataset = load_dataset("json", data_files=data_files)

        # Device-specific training arguments
        training_args = self._get_training_args(eval_file is not None)

        # Create trainer (trl >= 0.26 API)
        trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("eval"),
            args=training_args,
            formatting_func=lambda x: format_alpaca_prompt(x),
        )

        # Train
        trainer.train()

        # Save the adapter
        adapter_path = Path(self.config.output_dir) / "adapter"
        trainer.save_model(str(adapter_path))

        return adapter_path

    def _get_training_args(self, has_eval: bool) -> SFTConfig:
        """Get device-specific training arguments."""
        # Base arguments
        args = {
            "output_dir": self.config.output_dir,
            "num_train_epochs": self.config.num_epochs,
            "per_device_train_batch_size": self.config.batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": self.config.warmup_ratio,
            "weight_decay": self.config.weight_decay,
            "logging_steps": self.config.logging_steps,
            "save_strategy": "steps",
            "save_steps": self.config.save_steps,
            "eval_strategy": "steps" if has_eval else "no",
            "eval_steps": self.config.save_steps if has_eval else None,
            "max_grad_norm": 0.3,
            "group_by_length": True,
            "report_to": "none",
            "dataloader_pin_memory": False,  # Required for MPS
            "max_length": self.config.max_seq_length,
            "packing": True,
        }

        # Device-specific settings
        if self.device_type == DeviceType.CUDA:
            if self.config.use_4bit:
                args["optim"] = "adamw_8bit"
                args["bf16"] = True
                args["fp16"] = False
            else:
                args["optim"] = "adamw_torch"
                args["fp16"] = True
                args["bf16"] = False
        elif self.device_type == DeviceType.MPS:
            # MPS settings
            args["optim"] = "adamw_torch"
            args["bf16"] = False  # MPS doesn't support bf16 well
            args["packing"] = False  # Disable packing on MPS (no flash attention)

            if self.config.is_moe:
                # MoE models on MPS: use fp16, smaller batch for memory
                args["fp16"] = True
                args["per_device_train_batch_size"] = 1
                args["gradient_accumulation_steps"] = max(
                    self.config.gradient_accumulation_steps, 8
                )
                # Reduce save frequency for MoE (larger checkpoints)
                args["save_steps"] = max(self.config.save_steps, 200)
            else:
                # Dense models on MPS: use fp32 for stability
                args["fp16"] = False
                args["per_device_train_batch_size"] = min(self.config.batch_size, 2)
                args["gradient_accumulation_steps"] = max(
                    self.config.gradient_accumulation_steps,
                    self.config.batch_size // 2,
                )
        else:
            # CPU settings
            args["optim"] = "adamw_torch"
            args["fp16"] = False
            args["bf16"] = False

        return SFTConfig(**args)

    def merge_and_save(self, adapter_path: str | Path, output_path: str | Path) -> None:
        """Merge LoRA adapter with base model and save."""
        from peft import PeftModel

        print(f"Merging adapter from {adapter_path}")

        # Determine dtype based on device
        if self.device_type == DeviceType.CUDA:
            dtype = torch.float16
        else:
            dtype = torch.float32

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=dtype,
            device_map="auto" if self.device_type == DeviceType.CUDA else "cpu",
            trust_remote_code=True,
        )

        # Load and merge adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
        merged_model = model.merge_and_unload()

        # Save
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(output_path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_path)

        print(f"Merged model saved to {output_path}")


# Backward compatibility alias
QLoRAConfig = LoRAConfig
