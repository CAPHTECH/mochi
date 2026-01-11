#!/usr/bin/env python3
"""Create mixed training dataset for Project Adapter.

Combines:
1. Common patterns (from Base Adapter training data)
2. Project-specific patterns (from project codebase)

This prevents catastrophic forgetting of base patterns during project fine-tuning.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file."""
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def save_jsonl(items: list[dict], path: Path):
    """Save to JSONL file."""
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def create_mixed_dataset(
    common_data_dir: Path,
    project_data_dir: Path,
    output_dir: Path,
    common_ratio: float = 0.3,
    max_project_samples: int | None = None,
):
    """Create mixed dataset with common patterns interspersed.

    Args:
        common_data_dir: Directory with common patterns (train.jsonl, valid.jsonl)
        project_data_dir: Directory with project data (train.jsonl, valid.jsonl)
        output_dir: Output directory for mixed dataset
        common_ratio: Ratio of common patterns to include (0.3 = 30% common, 70% project)
        max_project_samples: Maximum project samples to use (None = all)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    common_train = load_jsonl(common_data_dir / "train.jsonl")
    common_valid = load_jsonl(common_data_dir / "valid.jsonl")
    project_train = load_jsonl(project_data_dir / "train.jsonl")
    project_valid = load_jsonl(project_data_dir / "valid.jsonl")

    print(f"Common patterns: {len(common_train)} train, {len(common_valid)} valid")
    print(f"Project patterns: {len(project_train)} train, {len(project_valid)} valid")

    # Limit project samples if specified
    if max_project_samples:
        random.shuffle(project_train)
        project_train = project_train[:max_project_samples]
        project_valid = project_valid[:max(100, max_project_samples // 10)]
        print(f"After limiting: {len(project_train)} train, {len(project_valid)} valid")

    # Calculate how many common patterns to include
    # Formula: common_count / (common_count + project_count) = common_ratio
    # common_count = (common_ratio * project_count) / (1 - common_ratio)
    target_common_train = int((common_ratio * len(project_train)) / (1 - common_ratio))
    target_common_valid = int((common_ratio * len(project_valid)) / (1 - common_ratio))

    # Repeat common patterns if needed
    def repeat_to_count(items: list[dict], target: int) -> list[dict]:
        if len(items) >= target:
            random.shuffle(items)
            return items[:target]
        result = []
        while len(result) < target:
            random.shuffle(items)
            result.extend(items)
        return result[:target]

    common_train_selected = repeat_to_count(common_train.copy(), target_common_train)
    common_valid_selected = repeat_to_count(common_valid.copy(), target_common_valid)

    print(f"Selected common: {len(common_train_selected)} train, {len(common_valid_selected)} valid")

    # Mix and shuffle
    mixed_train = project_train + common_train_selected
    mixed_valid = project_valid + common_valid_selected

    random.shuffle(mixed_train)
    random.shuffle(mixed_valid)

    print(f"Final mixed: {len(mixed_train)} train, {len(mixed_valid)} valid")

    # Save
    save_jsonl(mixed_train, output_dir / "train.jsonl")
    save_jsonl(mixed_valid, output_dir / "valid.jsonl")

    # Calculate actual ratio
    actual_common = len(common_train_selected) / len(mixed_train)
    print(f"Actual common ratio: {actual_common:.1%}")

    return len(mixed_train), len(mixed_valid)


def main():
    project_root = Path(__file__).parent.parent

    common_data = project_root / "data" / "common-patterns"
    project_data = project_root / "data" / "kiri-project"
    output_data = project_root / "data" / "kiri-mixed"

    if not common_data.exists():
        print(f"ERROR: Common patterns not found at {common_data}")
        return 1

    if not project_data.exists():
        print(f"ERROR: Project data not found at {project_data}")
        return 1

    print("=" * 70)
    print("Creating Mixed Dataset")
    print("=" * 70)
    print()

    # Create mixed dataset
    # 30% common patterns, max 2000 project samples (to keep training fast)
    train_count, valid_count = create_mixed_dataset(
        common_data,
        project_data,
        output_data,
        common_ratio=0.3,
        max_project_samples=2000,
    )

    print()
    print("=" * 70)
    print(f"Mixed dataset created at: {output_data}")
    print(f"Train: {train_count}, Valid: {valid_count}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
