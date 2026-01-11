#!/usr/bin/env python3
"""Qwen3-Coder-30B-A3B LoRA training script for mochi.

Uses mlx-lm library for efficient fine-tuning on Apple Silicon.

Usage:
    python scripts/train_qwen3_coder_lora.py

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
    output_dir = project_root / "output" / "mlx-qwen3-coder"

    # Model - Qwen3-Coder-30B-A3B 4-bit quantized for MLX
    model = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"

    if not train_file.exists():
        print(f"ERROR: Training data not found at {train_file}")
        print("Run scripts/regenerate_training_data.py first")
        return 1

    print("=" * 70)
    print("Qwen3-Coder-30B-A3B LoRA Training with MLX")
    print("=" * 70)
    print(f"Model: {model}")
    print(f"Train: {train_file}")
    print(f"Valid: {valid_file}")
    print(f"Output: {output_dir}")
    print()

    # Note about memory
    print("Note: Qwen3-Coder-30B-A3B 4-bit requires ~20GB memory")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # MLX LoRA training command
    # Increased iterations from 200 to 500 for better API accuracy
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model,
        "--train",
        "--data", str(data_dir),
        "--iters", "500",
        "--batch-size", "1",
        "--num-layers", "8",
        "--adapter-path", str(output_dir / "adapter"),
        "--learning-rate", "1e-5",
        "--grad-checkpoint",
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
