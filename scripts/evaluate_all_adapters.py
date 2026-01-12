#!/usr/bin/env python3
"""Evaluate all adapters: Base, Project, and Mixed.

Compares performance across different training approaches.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Test cases for evaluation
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


def evaluate_adapter(model_name: str, adapter_path: Path | None, test_cases: list, label: str) -> dict:
    """Evaluate a single adapter configuration."""
    from mlx_lm import load, generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler

    print(f"\n{'=' * 60}")
    print(f"Evaluating: {label}")
    print(f"{'=' * 60}")

    if adapter_path:
        model, tokenizer = load(model_name, adapter_path=str(adapter_path))
    else:
        model, tokenizer = load(model_name)

    sampler = make_sampler(temp=0.1)

    results = []
    total_score = 0
    category_scores = {}

    for i, test_case in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] {test_case['name']}...", end=" ")
        sys.stdout.flush()

        prompt = f"### Instruction:\n{test_case['instruction']}\n\n### Input:\n{test_case['input']}\n\n### Response:\n"

        start_time = time.time()
        response = mlx_generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=512,
            sampler=sampler,
        )
        inference_time = time.time() - start_time

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
            "response": response[:300],
        })

        total_score += score
        cat = test_case["category"]
        if cat not in category_scores:
            category_scores[cat] = {"total": 0, "count": 0}
        category_scores[cat]["total"] += score
        category_scores[cat]["count"] += 1

        print(f"score={score:.2f} ({inference_time:.1f}s)")

    avg_score = total_score / len(test_cases)
    category_averages = {cat: data["total"] / data["count"] for cat, data in category_scores.items()}

    print(f"\n--- Summary: {label} ---")
    print(f"Overall: {avg_score:.2%}")
    for cat, score in category_averages.items():
        print(f"  {cat}: {score:.2%}")

    return {
        "label": label,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "avg_score": avg_score,
        "category_scores": category_averages,
        "results": results,
    }


def main():
    project_root = Path(__file__).parent.parent
    model_name = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"

    adapters = [
        (project_root / "output" / "base-adapter" / "adapter", "Base Adapter"),
        (project_root / "output" / "kiri-adapter" / "adapter", "Project Adapter (resume from base)"),
        (project_root / "output" / "kiri-adapter-mixed" / "adapter", "Mixed Adapter (30% common)"),
    ]

    print("=" * 70)
    print("All Adapters Comparison")
    print("=" * 70)

    all_results = []

    for adapter_path, label in adapters:
        if adapter_path.exists():
            result = evaluate_adapter(model_name, adapter_path, TEST_CASES, label)
            all_results.append(result)
        else:
            print(f"\nSkipping: {label} (not found at {adapter_path})")

    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print("\n{:<40} {:>10}".format("Adapter", "Score"))
    print("-" * 52)
    for r in all_results:
        print("{:<40} {:>10.1%}".format(r["label"], r["avg_score"]))

    print("\n\nBy Category:")
    categories = list(all_results[0]["category_scores"].keys()) if all_results else []

    # Header
    header = "{:<20}".format("Category")
    for r in all_results:
        header += " {:>15}".format(r["label"][:15])
    print(header)
    print("-" * (20 + 16 * len(all_results)))

    # Rows
    for cat in categories:
        row = "{:<20}".format(cat)
        for r in all_results:
            score = r["category_scores"].get(cat, 0)
            row += " {:>14.1%}".format(score)
        print(row)

    # Save results
    output_file = project_root / "output" / "all_adapters_comparison.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
