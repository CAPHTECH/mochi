#!/usr/bin/env python3
"""Create enhanced mixed training dataset for Project Adapter.

Combines:
1. Common patterns (from Base Adapter training data)
2. Kiri-specific patterns (curated patterns for kiri architecture)
3. Kiri project code (from codebase)

This approach teaches both general patterns and project-specific idioms.
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


def repeat_to_count(items: list[dict], target: int) -> list[dict]:
    """Repeat items to reach target count."""
    if len(items) >= target:
        shuffled = items.copy()
        random.shuffle(shuffled)
        return shuffled[:target]
    result = []
    while len(result) < target:
        shuffled = items.copy()
        random.shuffle(shuffled)
        result.extend(shuffled)
    return result[:target]


def create_enhanced_mixed_dataset(
    common_data_dir: Path,
    kiri_patterns_dir: Path,
    kiri_project_dir: Path,
    output_dir: Path,
    common_ratio: float = 0.25,
    kiri_pattern_ratio: float = 0.25,
    kiri_project_ratio: float = 0.50,
    max_total_samples: int = 3000,
):
    """Create enhanced mixed dataset with three data sources.

    Args:
        common_data_dir: Common patterns (error-handling, async/await, etc.)
        kiri_patterns_dir: Kiri-specific curated patterns
        kiri_project_dir: Kiri project code extractions
        output_dir: Output directory
        common_ratio: Ratio of common patterns (default 25%)
        kiri_pattern_ratio: Ratio of kiri-specific patterns (default 25%)
        kiri_project_ratio: Ratio of kiri project code (default 50%)
        max_total_samples: Maximum total training samples
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    common_train = load_jsonl(common_data_dir / "train.jsonl")
    common_valid = load_jsonl(common_data_dir / "valid.jsonl")

    kiri_pattern_train = load_jsonl(kiri_patterns_dir / "train.jsonl")
    kiri_pattern_valid = load_jsonl(kiri_patterns_dir / "valid.jsonl")

    kiri_project_train = load_jsonl(kiri_project_dir / "train.jsonl")
    kiri_project_valid = load_jsonl(kiri_project_dir / "valid.jsonl")

    print(f"Data sources:")
    print(f"  Common patterns: {len(common_train)} train, {len(common_valid)} valid")
    print(f"  Kiri patterns: {len(kiri_pattern_train)} train, {len(kiri_pattern_valid)} valid")
    print(f"  Kiri project: {len(kiri_project_train)} train, {len(kiri_project_valid)} valid")

    # Calculate target counts
    target_common = int(max_total_samples * common_ratio)
    target_kiri_pattern = int(max_total_samples * kiri_pattern_ratio)
    target_kiri_project = int(max_total_samples * kiri_project_ratio)

    print(f"\nTarget distribution:")
    print(f"  Common: {target_common} ({common_ratio:.0%})")
    print(f"  Kiri patterns: {target_kiri_pattern} ({kiri_pattern_ratio:.0%})")
    print(f"  Kiri project: {target_kiri_project} ({kiri_project_ratio:.0%})")

    # Select samples (repeat smaller datasets to reach target)
    common_selected = repeat_to_count(common_train, target_common)
    kiri_pattern_selected = repeat_to_count(kiri_pattern_train, target_kiri_pattern)

    # For project data, sample without exceeding available
    kiri_project_selected = kiri_project_train.copy()
    random.shuffle(kiri_project_selected)
    kiri_project_selected = kiri_project_selected[:target_kiri_project]

    # Combine and shuffle
    mixed_train = common_selected + kiri_pattern_selected + kiri_project_selected
    random.shuffle(mixed_train)

    # Validation set (smaller, proportional)
    valid_total = min(300, len(mixed_train) // 10)
    valid_common = repeat_to_count(common_valid, int(valid_total * common_ratio))
    valid_kiri_pattern = repeat_to_count(kiri_pattern_valid, int(valid_total * kiri_pattern_ratio))
    valid_kiri_project = kiri_project_valid[:int(valid_total * kiri_project_ratio)]

    mixed_valid = valid_common + valid_kiri_pattern + valid_kiri_project
    random.shuffle(mixed_valid)

    print(f"\nActual distribution:")
    print(f"  Common: {len(common_selected)} ({len(common_selected)/len(mixed_train):.1%})")
    print(f"  Kiri patterns: {len(kiri_pattern_selected)} ({len(kiri_pattern_selected)/len(mixed_train):.1%})")
    print(f"  Kiri project: {len(kiri_project_selected)} ({len(kiri_project_selected)/len(mixed_train):.1%})")
    print(f"\nFinal: {len(mixed_train)} train, {len(mixed_valid)} valid")

    # Save
    save_jsonl(mixed_train, output_dir / "train.jsonl")
    save_jsonl(mixed_valid, output_dir / "valid.jsonl")

    return len(mixed_train), len(mixed_valid)


def main():
    project_root = Path(__file__).parent.parent

    common_data = project_root / "data" / "common-patterns"
    kiri_patterns = project_root / "data" / "kiri-patterns"
    kiri_project = project_root / "data" / "kiri-project"
    output_data = project_root / "data" / "kiri-enhanced-mixed"

    # Verify all data sources exist
    missing = []
    for path, name in [(common_data, "common-patterns"), (kiri_patterns, "kiri-patterns"), (kiri_project, "kiri-project")]:
        if not path.exists():
            missing.append(name)

    if missing:
        print(f"ERROR: Missing data directories: {', '.join(missing)}")
        return 1

    print("=" * 70)
    print("Creating Enhanced Mixed Dataset")
    print("=" * 70)
    print()

    # Create enhanced mixed dataset
    # 25% common patterns + 25% kiri patterns + 50% kiri project code
    train_count, valid_count = create_enhanced_mixed_dataset(
        common_data,
        kiri_patterns,
        kiri_project,
        output_data,
        common_ratio=0.25,
        kiri_pattern_ratio=0.25,
        kiri_project_ratio=0.50,
        max_total_samples=3000,
    )

    print()
    print("=" * 70)
    print(f"Enhanced mixed dataset created at: {output_data}")
    print(f"Train: {train_count}, Valid: {valid_count}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
