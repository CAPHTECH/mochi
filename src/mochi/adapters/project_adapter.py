"""Project Adapter implementation for project-specific patterns.

ProjectAdapter is fine-tuned on project-specific patterns:
- Custom types and interfaces from LSP
- Project naming conventions
- Custom error classes
- Domain-specific utilities

ProjectAdapters build on top of BaseAdapters, combining common patterns
with project-specific knowledge.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.exceptions import AdapterError, AdapterNotFoundError, TrainingError
from ..core.types import ProjectAdapterConfig, TrainingConfig

if TYPE_CHECKING:
    from .base_adapter import BaseAdapter

logger = logging.getLogger(__name__)


class ProjectAdapter:
    """Project-specific adapter.

    ProjectAdapter wraps a LoRA adapter trained on project-specific patterns.
    It is designed to be stacked on top of a BaseAdapter for best results.

    Usage:
        # Load base adapter first
        base = BaseAdapter.load("mochi-base-ts-v1")

        # Load project adapter
        project = ProjectAdapter.load("my-project-adapter", base_adapter=base)

        # Or create from LSP extraction
        project = ProjectAdapter.from_lsp(
            project_root=Path("/path/to/project"),
            base_adapter=base,
        )

        # Use for inference (uses stacked adapters)
        result = project.generate(prompt, max_tokens=256)
    """

    def __init__(
        self,
        config: ProjectAdapterConfig,
        base_adapter: BaseAdapter | None = None,
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        """Initialize ProjectAdapter.

        Args:
            config: Adapter configuration
            base_adapter: Optional base adapter to stack on
            model: Loaded MLX model (optional, lazy loaded if not provided)
            tokenizer: Tokenizer for the model (optional, lazy loaded)
        """
        self.config = config
        self.base_adapter = base_adapter
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
    def project_root(self) -> Path | None:
        """Get project root path."""
        return self.config.project_root

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
            logger.info(f"Loaded ProjectAdapter '{self.name}' from {adapter_path}")

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
            "base_adapter": self.config.base_adapter,
            "project_root": str(self.config.project_root) if self.config.project_root else None,
            "languages": self.config.languages,
            "include_patterns": self.config.include_patterns,
            "exclude_patterns": self.config.exclude_patterns,
            "version": self.config.version,
            "description": self.config.description,
            "metadata": self.config.metadata,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        # Update config path
        self.config.adapter_path = path

        logger.info(f"Saved ProjectAdapter config to {config_path}")

    @classmethod
    def load(
        cls,
        path: Path | str,
        base_adapter: BaseAdapter | None = None,
        lazy: bool = True,
    ) -> ProjectAdapter:
        """Load adapter from disk.

        Args:
            path: Path to adapter directory
            base_adapter: Optional base adapter to stack on
            lazy: If True, defer model loading until first use

        Returns:
            Loaded ProjectAdapter instance
        """
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

        config = ProjectAdapterConfig(
            name=config_data.get("name", path.name),
            adapter_type=config_data.get("adapter_type", "project"),
            base_model=config_data.get("base_model", ""),
            adapter_path=path,
            base_adapter=config_data.get("base_adapter"),
            project_root=Path(config_data["project_root"]) if config_data.get("project_root") else None,
            languages=config_data.get("languages", ["typescript"]),
            include_patterns=config_data.get("include_patterns", ["*.ts", "*.tsx"]),
            exclude_patterns=config_data.get("exclude_patterns", ["node_modules/**"]),
            version=config_data.get("version", "1.0.0"),
            description=config_data.get("description", ""),
            metadata=config_data.get("metadata", {}),
        )

        adapter = cls(config, base_adapter=base_adapter)

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
        base_adapter: BaseAdapter | None = None,
        training_config: TrainingConfig | None = None,
        project_root: Path | str | None = None,
        languages: list[str] | None = None,
    ) -> ProjectAdapter:
        """Train a new ProjectAdapter.

        Uses MLX-lm for efficient training on Apple Silicon.

        Args:
            name: Name for the adapter
            base_model: Base model name
            training_data: Path to training data directory (must contain train.jsonl)
            output_dir: Output directory for the adapter
            base_adapter: Optional base adapter to build on
            training_config: Training hyperparameters
            project_root: Project root path (for reference)
            languages: Languages this adapter supports

        Returns:
            Trained ProjectAdapter instance
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
            "--iters", str(training_config.epochs * 100),
            "--batch-size", str(training_config.batch_size),
            "--num-layers", str(training_config.lora_layers),
            "--adapter-path", str(adapter_path),
            "--learning-rate", str(training_config.learning_rate),
        ]

        # If base adapter exists, start from its weights
        if base_adapter and base_adapter.adapter_path:
            cmd.extend(["--resume-adapter-file", str(base_adapter.adapter_path)])

        if training_config.use_mps:
            cmd.append("--grad-checkpoint")

        logger.info(f"Training ProjectAdapter '{name}'")
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
        config = ProjectAdapterConfig(
            name=name,
            adapter_type="project",
            base_model=base_model,
            adapter_path=adapter_path,
            base_adapter=base_adapter.name if base_adapter else None,
            project_root=Path(project_root) if project_root else None,
            languages=languages or ["typescript"],
        )

        adapter = cls(config, base_adapter=base_adapter)
        adapter.save(output_dir)

        return adapter

    @classmethod
    def from_lsp(
        cls,
        project_root: Path | str,
        name: str | None = None,
        base_adapter: BaseAdapter | None = None,
        base_model: str = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
        output_dir: Path | str | None = None,
        languages: list[str] | None = None,
        training_config: TrainingConfig | None = None,
    ) -> ProjectAdapter:
        """Create ProjectAdapter from LSP context extraction.

        This method:
        1. Uses LSP to extract types, methods, and patterns from the project
        2. Generates training data from the extracted context
        3. Trains a new ProjectAdapter

        Args:
            project_root: Path to the project
            name: Name for the adapter (defaults to project directory name)
            base_adapter: Optional base adapter to build on
            base_model: Base model for training
            output_dir: Output directory (defaults to project_root/.mochi/adapter)
            languages: Languages to extract from
            training_config: Training hyperparameters

        Returns:
            Trained ProjectAdapter instance
        """
        from ..lsp.context_extractor import LSPContextExtractor

        project_root = Path(project_root).resolve()
        name = name or project_root.name
        output_dir = Path(output_dir) if output_dir else project_root / ".mochi" / "adapter"
        languages = languages or ["typescript"]

        logger.info(f"Creating ProjectAdapter from LSP for '{name}'")
        logger.info(f"Project root: {project_root}")

        # Step 1: Extract context from LSP
        data_dir = output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        extractor = LSPContextExtractor(project_root)
        training_examples = extractor.generate_training_data(
            languages=languages,
            output_dir=data_dir,
        )

        logger.info(f"Generated {len(training_examples)} training examples from LSP")

        # Step 2: Train the adapter
        return cls.train(
            name=name,
            base_model=base_model,
            training_data=data_dir,
            output_dir=output_dir,
            base_adapter=base_adapter,
            training_config=training_config,
            project_root=project_root,
            languages=languages,
        )

    def __repr__(self) -> str:
        base = f", base='{self.config.base_adapter}'" if self.config.base_adapter else ""
        return f"ProjectAdapter(name='{self.name}'{base})"
