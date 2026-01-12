#!/usr/bin/env python3
"""Comprehensive evaluation with both common and project-specific patterns.

Tests:
1. Common patterns (error-handling, async/await, type-safety, null-safety, validation)
2. Kiri-specific patterns (registry, typed protocol, config validation, etc.)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Common patterns test cases
COMMON_TEST_CASES = [
    {
        "name": "error-handling",
        "category": "common",
        "instruction": "Add error handling to this function",
        "input": """async function fetchData(url: string) {
  const response = await fetch(url);
  return await response.json();
}""",
        "expected_patterns": ["try", "catch", "throw", "Error"],
    },
    {
        "name": "async-await",
        "category": "common",
        "instruction": "Convert this callback-based code to async/await",
        "input": """function loadUser(id, callback) {
  fetch('/api/users/' + id)
    .then(res => res.json())
    .then(data => callback(null, data))
    .catch(err => callback(err));
}""",
        "expected_patterns": ["async", "await", "Promise"],
    },
    {
        "name": "type-safety",
        "category": "common",
        "instruction": "Add type annotations to this code",
        "input": """function greet(name, age) {
  return "Hello " + name + ", you are " + age + " years old";
}""",
        "expected_patterns": ["string", "number", ": "],
    },
    {
        "name": "null-safety",
        "category": "common",
        "instruction": "Add null safety checks to this code",
        "input": """function getUserEmail(user) {
  return user.profile.email;
}""",
        "expected_patterns": ["?.", "??", "undefined", "null"],
    },
    {
        "name": "validation",
        "category": "common",
        "instruction": "Add input validation to this function",
        "input": """function createUser(data) {
  return db.users.create(data);
}""",
        "expected_patterns": ["z.", "parse", "schema", "valid"],
    },
]

# Kiri-specific patterns test cases
KIRI_TEST_CASES = [
    {
        "name": "singleton-registry",
        "category": "kiri",
        "instruction": "Implement a singleton registry pattern for managing language analyzers",
        "input": """// Create a LanguageRegistry class that:
// - Is a singleton (getInstance)
// - Stores analyzers in a Map
// - Has register() and analyze() methods
interface LanguageAnalyzer {
  language: string;
  analyze(context: AnalysisContext): Promise<AnalysisResult>;
  dispose?(): Promise<void>;
}""",
        "expected_patterns": ["getInstance", "Map<", "private static", "singleton"],
    },
    {
        "name": "discriminated-union",
        "category": "kiri",
        "instruction": "Create discriminated union types for content overlay changes",
        "input": """// Create types for:
// - AddContentOverlay with type: "add" and content: string
// - RemoveContentOverlay with type: "remove"
// - ContentOverlayChange as union of both""",
        "expected_patterns": ['type: "add"', 'type: "remove"', "ContentOverlayChange", "|"],
    },
    {
        "name": "weighted-scoring",
        "category": "kiri",
        "instruction": "Implement a weighted profile selection function based on keyword matching",
        "input": """interface ProfilePattern {
  profile: string;
  keywords: string[];
  weight: number;
}

const patterns: ProfilePattern[] = [
  { profile: "testfail", keywords: ["test fail", "failing test"], weight: 10 },
  { profile: "typeerror", keywords: ["type error", "type mismatch"], weight: 11 },
];

// Implement selectProfileFromQuery(query: string): string""",
        "expected_patterns": ["match", "weight", "score", "toLowerCase"],
    },
    {
        "name": "config-validation",
        "category": "kiri",
        "instruction": "Add validation for scoring weights configuration with backward compatibility",
        "input": """interface ScoringWeights {
  textMatch: number;
  pathMatch: number;
  graphInbound?: number; // new field, optional for backward compat
}

function validateWeights(weights: unknown): ScoringWeights {
  // Validate that weights is an object
  // Provide default for graphInbound if missing
  // Validate all numbers are finite and >= 0
}""",
        "expected_patterns": ["typeof", "undefined", "isFinite", "throw"],
    },
    {
        "name": "path-multipliers",
        "category": "kiri",
        "instruction": "Implement path-based boost multipliers sorted by prefix length",
        "input": """interface PathMultiplier {
  prefix: string;
  multiplier: number;
}

// Create path multipliers for:
// - "src/vs/workbench/contrib/" -> 2.4
// - "src/vs/workbench/" -> 2.2
// - "src/vs/" -> 1.8
// - "src/" -> 1.0
// Note: longer prefixes should be checked first""",
        "expected_patterns": ["prefix", "multiplier", "startsWith", "src/"],
    },
]

ALL_TEST_CASES = COMMON_TEST_CASES + KIRI_TEST_CASES


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
        cat = test_case["category"]
        print(f"[{i+1}/{len(test_cases)}] [{cat}] {test_case['name']}...", end=" ")
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
            "matches": matches,
            "total": len(test_case["expected_patterns"]),
            "response": response[:400],
        })

        total_score += score
        if cat not in category_scores:
            category_scores[cat] = {"total": 0, "count": 0}
        category_scores[cat]["total"] += score
        category_scores[cat]["count"] += 1

        print(f"score={score:.2f} ({matches}/{len(test_case['expected_patterns'])}) ({inference_time:.1f}s)")

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
        (project_root / "output" / "kiri-adapter" / "adapter", "Project Adapter (resume)"),
        (project_root / "output" / "kiri-adapter-mixed" / "adapter", "Mixed Adapter (30% common)"),
        (project_root / "output" / "kiri-adapter-enhanced" / "adapter", "Enhanced Mixed (25%+25%+50%)"),
    ]

    print("=" * 70)
    print("Comprehensive Evaluation: Common + Kiri-Specific Patterns")
    print("=" * 70)
    print(f"Common test cases: {len(COMMON_TEST_CASES)}")
    print(f"Kiri-specific test cases: {len(KIRI_TEST_CASES)}")
    print(f"Total: {len(ALL_TEST_CASES)}")

    all_results = []

    for adapter_path, label in adapters:
        if adapter_path.exists():
            result = evaluate_adapter(model_name, adapter_path, ALL_TEST_CASES, label)
            all_results.append(result)
        else:
            print(f"\nSkipping: {label} (not found)")

    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print("\n{:<35} {:>8} {:>8} {:>8}".format("Adapter", "Overall", "Common", "Kiri"))
    print("-" * 63)
    for r in all_results:
        common_score = r["category_scores"].get("common", 0)
        kiri_score = r["category_scores"].get("kiri", 0)
        print("{:<35} {:>7.1%} {:>7.1%} {:>7.1%}".format(
            r["label"][:35], r["avg_score"], common_score, kiri_score
        ))

    # Detailed breakdown
    print("\n\nDetailed Results:")
    print("-" * 70)

    for test_case in ALL_TEST_CASES:
        print(f"\n{test_case['name']} [{test_case['category']}]:")
        for r in all_results:
            test_result = next((t for t in r["results"] if t["name"] == test_case["name"]), None)
            if test_result:
                print(f"  {r['label'][:25]:<25}: {test_result['score']:.0%} ({test_result['matches']}/{test_result['total']})")

    # Save results
    output_file = project_root / "output" / "comprehensive_evaluation.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nResults saved to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
