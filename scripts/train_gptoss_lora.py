#!/usr/bin/env python3
"""GPT-OSS-20B LoRA training script for mochi.

Uses mlx-lm library for efficient fine-tuning on Apple Silicon.

Usage:
    python scripts/train_gptoss_lora.py

This will train using the LSP-enhanced training data.
"""

import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).parent.parent

    # Training data with LSP context
    data_dir = project_root / "data" / "mlx_lsp"
    train_file = data_dir / "train.jsonl"
    valid_file = data_dir / "valid.jsonl"
    output_dir = project_root / "output" / "gptoss-20b-lsp"

    # Model - use MLX-optimized gpt-oss-20b
    # Using 8-bit version for better quality (~20GB, requires ~32GB RAM)
    model = "lmstudio-community/gpt-oss-20b-MLX-8bit"

    if not train_file.exists():
        print(f"ERROR: Training data not found at {train_file}")
        print("Run scripts/regenerate_training_data.py first")
        return 1

    print("=" * 70)
    print("GPT-OSS-20B LoRA Training with MLX")
    print("=" * 70)
    print(f"Model: {model}")
    print(f"Train: {train_file}")
    print(f"Valid: {valid_file}")
    print(f"Output: {output_dir}")
    print()

    # Check memory
    print("Note: gpt-oss-20b requires ~16GB memory for 4-bit, ~32GB for 8-bit")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # MLX LoRA training command
    # Using reduced parameters due to model size
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model,
        "--train",
        "--data", str(data_dir),
        "--iters", "200",  # Reduced iterations for large model
        "--batch-size", "1",  # Minimal batch size due to memory
        "--num-layers", "8",  # Fewer LoRA layers
        "--adapter-path", str(output_dir / "adapter"),
        "--learning-rate", "1e-5",  # Lower learning rate for large model
        "--grad-checkpoint",  # Enable gradient checkpointing to save memory
    ]

    print("Running command:")
    print(" ".join(cmd))
    print()

    try:
        result = subprocess.run(cmd, check=True)
        print()
        print("=" * 70)
        print("Training complete!")
        print(f"Adapter saved to: {output_dir / 'adapter'}")
        print("=" * 70)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code: {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print("mlx-lm not installed. Install with: pip install mlx-lm")
        return 1


if __name__ == "__main__":
    sys.exit(main())
