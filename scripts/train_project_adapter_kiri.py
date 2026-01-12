#!/usr/bin/env python3
"""Train Project Adapter on kiri codebase.

Uses the Base Adapter as foundation and adds kiri-specific patterns.

Usage:
    python scripts/train_project_adapter_kiri.py
"""

import subprocess
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def extract_training_data(project_root: Path, output_dir: Path) -> tuple[int, int]:
    """Extract training data from kiri project."""
    from mochi.data_generation.alpaca_converter import create_training_dataset
    from mochi.ingestion.git_connector import GitConnector
    from mochi.preprocessing.code_chunker import ChunkStrategy, CodeChunker

    print(f"Extracting training data from: {project_root}")

    # Connect to repo
    connector = GitConnector(str(project_root))

    # Get TypeScript files
    files = connector.get_source_files([".ts", ".tsx"])
    print(f"Found {len(files)} TypeScript files")

    # Chunk code
    chunker = CodeChunker()
    all_chunks = []

    for source_file in files:
        try:
            chunks = chunker.chunk(
                source_file.path,
                source_file.content,
                source_file.language,
                strategy=ChunkStrategy.TOPLEVEL,
            )
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Warning: Failed to chunk {source_file.path}: {e}")

    print(f"Created {len(all_chunks)} chunks")

    # Convert to training format
    train_path, eval_path = create_training_dataset(
        all_chunks,
        output_dir,
        project_name="kiri",
    )

    # Count examples
    with open(train_path) as f:
        train_count = sum(1 for _ in f)
    with open(eval_path) as f:
        eval_count = sum(1 for _ in f)

    print(f"Train: {train_count} examples")
    print(f"Eval: {eval_count} examples")

    return train_count, eval_count


def main():
    project_root = Path(__file__).parent.parent
    kiri_root = Path.home() / "Workspace" / "kiri"

    if not kiri_root.exists():
        print(f"ERROR: kiri project not found at {kiri_root}")
        return 1

    # Output directories
    data_dir = project_root / "data" / "kiri-project"
    output_dir = project_root / "output" / "kiri-adapter"
    base_adapter_dir = project_root / "output" / "base-adapter" / "adapter"

    # Check base adapter exists
    if not base_adapter_dir.exists():
        print(f"ERROR: Base adapter not found at {base_adapter_dir}")
        print("Run scripts/train_base_adapter.py first")
        return 1

    # Model
    model = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"

    print("=" * 70)
    print("Project Adapter Training (kiri)")
    print("=" * 70)
    print(f"Project: {kiri_root}")
    print(f"Base Adapter: {base_adapter_dir}")
    print(f"Model: {model}")
    print()

    # Step 1: Extract training data
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        train_count, eval_count = extract_training_data(kiri_root, data_dir)
    except Exception as e:
        print(f"ERROR: Failed to extract training data: {e}")
        import traceback
        traceback.print_exc()
        return 1

    if train_count < 10:
        print(f"ERROR: Too few training examples ({train_count})")
        return 1

    # Step 2: Train Project Adapter
    # Using the base adapter as initialization
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("Training Project Adapter...")
    print()

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model,
        "--train",
        "--data", str(data_dir),
        "--iters", "200",  # Fewer iterations since base is pre-trained
        "--batch-size", "4",
        "--num-layers", "16",
        "--adapter-path", str(output_dir / "adapter"),
        "--learning-rate", "5e-6",  # Lower LR for fine-tuning
        "--resume-adapter-file", str(base_adapter_dir / "adapters.safetensors"),  # Start from base
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
            "name": "kiri-project",
            "type": "project",
            "base_model": model,
            "base_adapter": str(base_adapter_dir),
            "project": "kiri",
            "languages": ["typescript"],
            "train_examples": train_count,
            "eval_examples": eval_count,
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
