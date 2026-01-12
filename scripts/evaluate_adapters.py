#!/usr/bin/env python3
"""Evaluate Base Adapter and Project Adapter performance.

Compares:
1. Base model only (no adapter)
2. Base Adapter only
3. Base + Project Adapter stack

Uses test cases from common patterns and kiri-specific patterns.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Test cases for evaluation
TEST_CASES = [
    # Error handling patterns (Base Adapter should excel)
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


def load_model_and_adapter(model_name: str, adapter_path: Path | None = None):
    """Load model with optional adapter."""
    from mlx_lm import load, generate as mlx_generate

    if adapter_path and adapter_path.exists():
        model, tokenizer = load(model_name, adapter_path=str(adapter_path))
    else:
        model, tokenizer = load(model_name)

    return model, tokenizer


def generate_completion(model, tokenizer, instruction: str, input_text: str, max_tokens: int = 512) -> str:
    """Generate completion using the model."""
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler

    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"

    # Create sampler with low temperature for deterministic output
    sampler = make_sampler(temp=0.1)

    response = mlx_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    )

    # Extract response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    return response


def evaluate_response(response: str, expected_patterns: list[str]) -> dict:
    """Evaluate response against expected patterns."""
    response_lower = response.lower()

    matches = []
    misses = []

    for pattern in expected_patterns:
        if pattern.lower() in response_lower:
            matches.append(pattern)
        else:
            misses.append(pattern)

    score = len(matches) / len(expected_patterns) if expected_patterns else 0

    return {
        "score": score,
        "matches": matches,
        "misses": misses,
        "response_length": len(response),
    }


def run_evaluation(model_name: str, adapter_path: Path | None, test_cases: list[dict], label: str) -> dict:
    """Run evaluation on all test cases."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {label}")
    print(f"{'=' * 60}")

    if adapter_path:
        print(f"Adapter: {adapter_path}")
    print(f"Model: {model_name}")
    print()

    model, tokenizer = load_model_and_adapter(model_name, adapter_path)

    results = []
    total_score = 0
    category_scores = {}

    for i, test_case in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] {test_case['name']}...", end=" ")
        sys.stdout.flush()

        start_time = time.time()
        response = generate_completion(
            model, tokenizer,
            test_case["instruction"],
            test_case["input"],
        )
        inference_time = time.time() - start_time

        eval_result = evaluate_response(response, test_case["expected_patterns"])
        eval_result["name"] = test_case["name"]
        eval_result["category"] = test_case["category"]
        eval_result["inference_time"] = inference_time
        eval_result["response"] = response[:500]  # Truncate for display

        results.append(eval_result)
        total_score += eval_result["score"]

        # Category tracking
        cat = test_case["category"]
        if cat not in category_scores:
            category_scores[cat] = {"total": 0, "count": 0}
        category_scores[cat]["total"] += eval_result["score"]
        category_scores[cat]["count"] += 1

        print(f"score={eval_result['score']:.2f} ({inference_time:.1f}s)")

    avg_score = total_score / len(test_cases) if test_cases else 0

    # Calculate category averages
    category_averages = {}
    for cat, data in category_scores.items():
        category_averages[cat] = data["total"] / data["count"] if data["count"] > 0 else 0

    print(f"\n--- Summary for {label} ---")
    print(f"Average Score: {avg_score:.2%}")
    print("By Category:")
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

    base_adapter = project_root / "output" / "base-adapter" / "adapter"
    project_adapter = project_root / "output" / "kiri-adapter" / "adapter"

    print("=" * 70)
    print("Adapter Evaluation")
    print("=" * 70)

    all_results = []

    # 1. Base model only
    # result_base = run_evaluation(model_name, None, TEST_CASES, "Base Model (no adapter)")
    # all_results.append(result_base)

    # 2. Base Adapter only
    if base_adapter.exists():
        result_base_adapter = run_evaluation(model_name, base_adapter, TEST_CASES, "Base Adapter")
        all_results.append(result_base_adapter)
    else:
        print(f"Warning: Base adapter not found at {base_adapter}")

    # 3. Project Adapter (includes base patterns due to continued training)
    if project_adapter.exists():
        result_project = run_evaluation(model_name, project_adapter, TEST_CASES, "Project Adapter (kiri)")
        all_results.append(result_project)
    else:
        print(f"Warning: Project adapter not found at {project_adapter}")

    # Final comparison
    print("\n" + "=" * 70)
    print("Final Comparison")
    print("=" * 70)

    for result in all_results:
        print(f"\n{result['label']}:")
        print(f"  Overall: {result['avg_score']:.2%}")
        for cat, score in result["category_scores"].items():
            print(f"  {cat}: {score:.2%}")

    # Save results
    output_file = project_root / "output" / "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
