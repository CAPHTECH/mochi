#!/usr/bin/env python3
"""Base Adapter LoRA training script.

Trains a Base Adapter on common TypeScript/JavaScript patterns:
- error-handling
- async-await
- type-safety
- null-safety
- validation

Usage:
    python scripts/train_base_adapter.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).parent.parent

    # Training data
    data_dir = project_root / "data" / "common-patterns"
    train_file = data_dir / "train.jsonl"
    valid_file = data_dir / "valid.jsonl"
    output_dir = project_root / "output" / "base-adapter"

    # Model - Qwen3-Coder is optimized for code completion
    model = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"

    if not train_file.exists():
        print(f"ERROR: Training data not found at {train_file}")
        print("Run scripts/collect_base_patterns.py first")
        return 1

    # Count examples
    with open(train_file) as f:
        train_count = sum(1 for _ in f)
    with open(valid_file) as f:
        valid_count = sum(1 for _ in f)

    print("=" * 70)
    print("Base Adapter LoRA Training with MLX")
    print("=" * 70)
    print(f"Model: {model}")
    print(f"Train: {train_file} ({train_count} examples)")
    print(f"Valid: {valid_file} ({valid_count} examples)")
    print(f"Output: {output_dir}")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # MLX LoRA training command
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model,
        "--train",
        "--data", str(data_dir),
        "--iters", "300",  # More iterations for pattern learning
        "--batch-size", "4",  # Reasonable batch size for small model
        "--num-layers", "16",  # More LoRA layers for better learning
        "--adapter-path", str(output_dir / "adapter"),
        "--learning-rate", "1e-5",
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

        # Save adapter metadata
        metadata = {
            "name": "mochi-base",
            "type": "base",
            "base_model": model,
            "patterns": ["error-handling", "async-await", "type-safety", "null-safety", "validation"],
            "languages": ["typescript", "javascript"],
            "train_examples": train_count,
            "valid_examples": valid_count,
        }
        import json
        with open(output_dir / "adapter_config.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {output_dir / 'adapter_config.json'}")

        return 0
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code: {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print("mlx-lm not installed. Install with: pip install mlx-lm")
        return 1


if __name__ == "__main__":
    sys.exit(main())
