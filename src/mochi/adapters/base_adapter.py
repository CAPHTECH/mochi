"""Base Adapter implementation for common code patterns.

BaseAdapter is pre-trained on common TypeScript/JavaScript patterns:
- error-handling (try-catch, Result types)
- async-await (Promise handling)
- type-safety (type annotations, guards)
- null-safety (optional chaining, nullish coalescing)
- validation (zod, assertions)

These adapters can be shared across projects and used as a foundation
for project-specific adapters.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..core.exceptions import AdapterError, AdapterNotFoundError, TrainingError
from ..core.types import BaseAdapterConfig, TrainingConfig

logger = logging.getLogger(__name__)


class BaseAdapter:
    """Base adapter for common code patterns.

    BaseAdapter wraps a LoRA adapter trained on common TypeScript/JavaScript
    patterns. It can be loaded from disk or trained from data.

    Usage:
        # Load pre-trained adapter
        adapter = BaseAdapter.load("mochi-base-ts-v1")

        # Train new adapter
        adapter = BaseAdapter.train(
            base_model="mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
            training_data=Path("data/common-patterns"),
            output_dir=Path("output/my-base-adapter"),
        )

        # Use for inference
        result = adapter.generate(prompt, max_tokens=256)
    """

    def __init__(
        self,
        config: BaseAdapterConfig,
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        """Initialize BaseAdapter.

        Args:
            config: Adapter configuration
            model: Loaded MLX model (optional, lazy loaded if not provided)
            tokenizer: Tokenizer for the model (optional, lazy loaded)
        """
        self.config = config
        self._model = model
        self._tokenizer = tokenizer
        self._loaded = model is not None

    @property
    def name(self) -> str:
        """Get adapter name."""
        return self.config.name

    @property
    def adapter_path(self) -> Path | None:
        """Get adapter path."""
        return self.config.adapter_path

    @property
    def is_loaded(self) -> bool:
        """Check if adapter model is loaded."""
        return self._loaded

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if not self._loaded:
            self._load_model()

    def _load_model(self) -> None:
        """Load MLX model with adapter."""
        if self.config.adapter_path is None:
            raise AdapterError(
                "Cannot load model: adapter_path is not set",
                {"adapter_name": self.name},
            )

        try:
            from mlx_lm import load

            adapter_path = self.config.adapter_path

            # MLX-lm loads base model + adapter
            self._model, self._tokenizer = load(
                self.config.base_model,
                adapter_path=str(adapter_path),
            )
            self._loaded = True
            logger.info(f"Loaded BaseAdapter '{self.name}' from {adapter_path}")

        except ImportError as e:
            raise AdapterError(
                "mlx-lm is required for inference. Install with: pip install mlx-lm",
                {"error": str(e)},
            ) from e
        except Exception as e:
            raise AdapterError(
                f"Failed to load adapter from {self.config.adapter_path}",
                {"error": str(e)},
            ) from e

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> str:
        """Generate text using the adapter.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty

        Returns:
            Generated text
        """
        self._ensure_loaded()

        try:
            from mlx_lm import generate

            result = generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            return result

        except Exception as e:
            raise AdapterError(
                "Generation failed",
                {"error": str(e), "adapter": self.name},
            ) from e

    def save(self, path: Path | str) -> None:
        """Save adapter to disk.

        Args:
            path: Output directory for the adapter
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = path / "adapter_config.json"
        config_data = {
            "name": self.config.name,
            "adapter_type": self.config.adapter_type.value,
            "base_model": self.config.base_model,
            "patterns": self.config.patterns,
            "languages": self.config.languages,
            "version": self.config.version,
            "description": self.config.description,
            "metadata": self.config.metadata,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        # Update config path
        self.config.adapter_path = path

        logger.info(f"Saved BaseAdapter config to {config_path}")

    @classmethod
    def _download_from_hub(cls, repo_id: str, cache_dir: Path | None = None) -> Path:
        """Download adapter from HuggingFace Hub.

        Args:
            repo_id: HuggingFace Hub repo ID (e.g., "CAPHTECH/mochi-base-ts-v1")
            cache_dir: Local cache directory (default: ~/.cache/mochi/adapters)

        Returns:
            Path to downloaded adapter directory
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise AdapterError(
                "huggingface_hub is required for downloading adapters. "
                "Install with: pip install huggingface_hub",
                {"error": str(e)},
            ) from e

        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "mochi" / "adapters"

        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading adapter from HuggingFace Hub: {repo_id}")

        try:
            local_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=str(cache_dir),
                local_dir=cache_dir / repo_id.replace("/", "--"),
            )
            return Path(local_path)
        except Exception as e:
            raise AdapterError(
                f"Failed to download adapter from HuggingFace Hub: {repo_id}",
                {"error": str(e)},
            ) from e

    @classmethod
    def _is_hub_repo_id(cls, path: str) -> bool:
        """Check if path looks like a HuggingFace Hub repo ID."""
        # Hub repo IDs have format "org/repo" or "user/repo"
        # Local paths have "/" but also have "." or start with "/" or "~"
        if "/" not in path:
            return False
        if path.startswith(("/", "~", ".")):
            return False
        if "\\" in path:  # Windows path
            return False
        parts = path.split("/")
        return len(parts) == 2 and all(p and not p.startswith(".") for p in parts)

    @classmethod
    def load(
        cls,
        path: Path | str,
        lazy: bool = True,
    ) -> BaseAdapter:
        """Load adapter from disk or HuggingFace Hub.

        Args:
            path: Path to adapter directory or HuggingFace Hub repo ID
                  (e.g., "CAPHTECH/mochi-base-ts-v1")
            lazy: If True, defer model loading until first use

        Returns:
            Loaded BaseAdapter instance
        """
        path_str = str(path)

        # Check if it's a HuggingFace Hub repo ID
        if cls._is_hub_repo_id(path_str):
            path = cls._download_from_hub(path_str)
        else:
            path = Path(path)

        if not path.exists():
            raise AdapterNotFoundError(str(path), [str(path)])

        config_path = path / "adapter_config.json"
        if not config_path.exists():
            raise AdapterError(
                f"Adapter config not found at {config_path}",
                {"path": str(path)},
            )

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        config = BaseAdapterConfig(
            name=config_data.get("name", path.name),
            adapter_type=config_data.get("adapter_type", "base"),
            base_model=config_data.get("base_model", ""),
            adapter_path=path,
            patterns=config_data.get("patterns", []),
            languages=config_data.get("languages", ["typescript"]),
            version=config_data.get("version", "1.0.0"),
            description=config_data.get("description", ""),
            metadata=config_data.get("metadata", {}),
        )

        adapter = cls(config)

        if not lazy:
            adapter._load_model()

        return adapter

    @classmethod
    def train(
        cls,
        name: str,
        base_model: str,
        training_data: Path | str,
        output_dir: Path | str,
        training_config: TrainingConfig | None = None,
        patterns: list[str] | None = None,
        languages: list[str] | None = None,
    ) -> BaseAdapter:
        """Train a new BaseAdapter.

        Uses MLX-lm for efficient training on Apple Silicon.

        Args:
            name: Name for the adapter
            base_model: Base model name (e.g., "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit")
            training_data: Path to training data directory (must contain train.jsonl)
            output_dir: Output directory for the adapter
            training_config: Training hyperparameters
            patterns: Patterns this adapter is trained on
            languages: Languages this adapter supports

        Returns:
            Trained BaseAdapter instance
        """
        training_data = Path(training_data)
        output_dir = Path(output_dir)
        training_config = training_config or TrainingConfig()

        # Validate training data
        train_file = training_data / "train.jsonl"
        if not train_file.exists():
            raise TrainingError(
                f"Training file not found: {train_file}",
                {"training_data": str(training_data)},
            )

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        adapter_path = output_dir / "adapter"

        # Build MLX-lm training command
        cmd = [
            sys.executable, "-m", "mlx_lm", "lora",
            "--model", base_model,
            "--train",
            "--data", str(training_data),
            "--iters", str(training_config.epochs * 100),  # Approximate iterations
            "--batch-size", str(training_config.batch_size),
            "--num-layers", str(training_config.lora_layers),
            "--adapter-path", str(adapter_path),
            "--learning-rate", str(training_config.learning_rate),
        ]

        if training_config.use_mps:
            cmd.append("--grad-checkpoint")

        logger.info(f"Training BaseAdapter '{name}'")
        logger.info(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("Training completed successfully")
            logger.debug(result.stdout)

        except subprocess.CalledProcessError as e:
            raise TrainingError(
                "Training failed",
                {"exit_code": e.returncode, "stderr": e.stderr},
            ) from e
        except FileNotFoundError as e:
            raise TrainingError(
                "mlx-lm not installed. Install with: pip install mlx-lm",
                {"error": str(e)},
            ) from e

        # Create adapter config
        config = BaseAdapterConfig(
            name=name,
            adapter_type="base",
            base_model=base_model,
            adapter_path=adapter_path,
            patterns=patterns or [
                "error-handling",
                "async-await",
                "type-safety",
                "null-safety",
                "validation",
            ],
            languages=languages or ["typescript", "javascript"],
        )

        adapter = cls(config)
        adapter.save(output_dir)

        return adapter

    def __repr__(self) -> str:
        return f"BaseAdapter(name='{self.name}', patterns={self.config.patterns})"
