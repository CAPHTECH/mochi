#!/usr/bin/env python3
"""Train Project Adapter on mixed dataset.

Combines common patterns with project-specific patterns to prevent
catastrophic forgetting during fine-tuning.

Usage:
    python scripts/train_project_adapter_mixed.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).parent.parent

    # Training data
    data_dir = project_root / "data" / "kiri-mixed"
    output_dir = project_root / "output" / "kiri-adapter-mixed"

    # Model
    model = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"

    if not (data_dir / "train.jsonl").exists():
        print(f"ERROR: Mixed data not found at {data_dir}")
        print("Run scripts/create_mixed_dataset.py first")
        return 1

    # Count examples
    with open(data_dir / "train.jsonl") as f:
        train_count = sum(1 for _ in f)
    with open(data_dir / "valid.jsonl") as f:
        valid_count = sum(1 for _ in f)

    print("=" * 70)
    print("Project Adapter Training (Mixed Data)")
    print("=" * 70)
    print(f"Model: {model}")
    print(f"Data: {data_dir}")
    print(f"Train: {train_count} examples (30% common, 70% project)")
    print(f"Valid: {valid_count} examples")
    print(f"Output: {output_dir}")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # MLX LoRA training command
    # Training from scratch (not resuming from base) since data already includes common patterns
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model,
        "--train",
        "--data", str(data_dir),
        "--iters", "500",  # More iterations for mixed data
        "--batch-size", "4",
        "--num-layers", "16",
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

        # Save metadata
        import json
        metadata = {
            "name": "kiri-project-mixed",
            "type": "project-mixed",
            "base_model": model,
            "project": "kiri",
            "languages": ["typescript"],
            "train_examples": train_count,
            "valid_examples": valid_count,
            "common_ratio": 0.3,
            "approach": "mixed-data-training",
        }
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
