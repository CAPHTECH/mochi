#!/usr/bin/env python3
"""Evaluate AdapterStack combining Base and Project Adapters.

Tests different weight combinations to find optimal configuration.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Test cases
TEST_CASES = [
    # Error handling patterns
    {
        "name": "error-handling-1",
        "category": "error-handling",
        "instruction": "Add error handling to this function",
        "input": """async function fetchData(url: string) {
  const response = await fetch(url);
  return await response.json();
}""",
        "expected_patterns": ["try", "catch", "throw", "Error"],
    },
    {
        "name": "error-handling-2",
        "category": "error-handling",
        "instruction": "Wrap this code with try-catch",
        "input": """function parseJSON(text: string) {
  const data = JSON.parse(text);
  return data;
}""",
        "expected_patterns": ["try", "catch", "SyntaxError", "Error"],
    },
    # Async/await patterns
    {
        "name": "async-await-1",
        "category": "async-await",
        "instruction": "Convert this callback-based code to async/await",
        "input": """function loadUser(id, callback) {
  fetch('/api/users/' + id)
    .then(res => res.json())
    .then(data => callback(null, data))
    .catch(err => callback(err));
}""",
        "expected_patterns": ["async", "await", "Promise"],
    },
    # Type safety patterns
    {
        "name": "type-safety-1",
        "category": "type-safety",
        "instruction": "Add type annotations to this code",
        "input": """function greet(name, age) {
  return "Hello " + name + ", you are " + age + " years old";
}""",
        "expected_patterns": ["string", "number", ": "],
    },
    # Null safety patterns
    {
        "name": "null-safety-1",
        "category": "null-safety",
        "instruction": "Add null safety checks to this code",
        "input": """function getUserEmail(user) {
  return user.profile.email;
}""",
        "expected_patterns": ["?.", "??", "undefined", "null"],
    },
    # Validation patterns
    {
        "name": "validation-1",
        "category": "validation",
        "instruction": "Add input validation to this function",
        "input": """function createUser(data) {
  return db.users.create(data);
}""",
        "expected_patterns": ["z.", "parse", "schema", "valid"],
    },
]


def merge_adapters(base_path: Path, project_path: Path, output_path: Path, base_weight: float, project_weight: float):
    """Merge two adapters with specified weights."""
    import mlx.core as mx
    from safetensors.numpy import load_file, save_file
    import numpy as np

    base_weights = load_file(str(base_path / "adapters.safetensors"))
    project_weights = load_file(str(project_path / "adapters.safetensors"))

    merged = {}
    for key in base_weights:
        if key in project_weights:
            # Weighted average
            merged[key] = base_weights[key] * base_weight + project_weights[key] * project_weight
        else:
            merged[key] = base_weights[key]

    # Add any keys only in project
    for key in project_weights:
        if key not in merged:
            merged[key] = project_weights[key]

    output_path.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(output_path / "adapters.safetensors"))

    # Copy adapter_config.json from base
    import shutil
    if (base_path / "adapter_config.json").exists():
        shutil.copy(base_path / "adapter_config.json", output_path / "adapter_config.json")
    elif (base_path / "adapter_config.yaml").exists():
        shutil.copy(base_path / "adapter_config.yaml", output_path / "adapter_config.yaml")

    return output_path


def evaluate_with_weights(model_name: str, base_path: Path, project_path: Path,
                          base_weight: float, project_weight: float, test_cases: list) -> dict:
    """Evaluate merged adapter with specific weights."""
    from mlx_lm import load, generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler

    # Create merged adapter
    merged_path = Path(f"/tmp/merged_adapter_{base_weight}_{project_weight}")
    merge_adapters(base_path, project_path, merged_path, base_weight, project_weight)

    # Load model with merged adapter
    model, tokenizer = load(model_name, adapter_path=str(merged_path))
    sampler = make_sampler(temp=0.1)

    results = []
    total_score = 0
    category_scores = {}

    for test_case in test_cases:
        prompt = f"### Instruction:\n{test_case['instruction']}\n\n### Input:\n{test_case['input']}\n\n### Response:\n"

        response = mlx_generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=512,
            sampler=sampler,
        )

        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()

        # Score
        response_lower = response.lower()
        matches = sum(1 for p in test_case["expected_patterns"] if p.lower() in response_lower)
        score = matches / len(test_case["expected_patterns"])

        results.append({
            "name": test_case["name"],
            "category": test_case["category"],
            "score": score,
        })

        total_score += score
        cat = test_case["category"]
        if cat not in category_scores:
            category_scores[cat] = {"total": 0, "count": 0}
        category_scores[cat]["total"] += score
        category_scores[cat]["count"] += 1

    avg_score = total_score / len(test_cases)
    category_averages = {cat: data["total"] / data["count"] for cat, data in category_scores.items()}

    return {
        "base_weight": base_weight,
        "project_weight": project_weight,
        "avg_score": avg_score,
        "category_scores": category_averages,
        "results": results,
    }


def main():
    project_root = Path(__file__).parent.parent
    model_name = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit"

    base_adapter = project_root / "output" / "base-adapter" / "adapter"
    project_adapter = project_root / "output" / "kiri-adapter" / "adapter"

    if not base_adapter.exists() or not project_adapter.exists():
        print("ERROR: Adapters not found")
        return 1

    print("=" * 70)
    print("AdapterStack Weight Optimization")
    print("=" * 70)

    # Test different weight combinations
    weight_configs = [
        (1.0, 0.0),   # Base only
        (0.7, 0.3),   # More base
        (0.5, 0.5),   # Equal
        (0.3, 0.7),   # More project
        (0.0, 1.0),   # Project only
    ]

    all_results = []

    for base_w, proj_w in weight_configs:
        print(f"\nTesting: Base={base_w}, Project={proj_w}")
        result = evaluate_with_weights(
            model_name, base_adapter, project_adapter,
            base_w, proj_w, TEST_CASES
        )
        all_results.append(result)

        print(f"  Overall: {result['avg_score']:.2%}")
        for cat, score in result["category_scores"].items():
            print(f"  {cat}: {score:.2%}")

    # Find best configuration
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)

    best = max(all_results, key=lambda x: x["avg_score"])
    print(f"\nBest Configuration:")
    print(f"  Base Weight: {best['base_weight']}")
    print(f"  Project Weight: {best['project_weight']}")
    print(f"  Overall Score: {best['avg_score']:.2%}")

    print("\nAll Configurations:")
    for r in all_results:
        print(f"  Base={r['base_weight']}, Project={r['project_weight']}: {r['avg_score']:.2%}")

    # Save results
    output_file = project_root / "output" / "adapter_stack_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
