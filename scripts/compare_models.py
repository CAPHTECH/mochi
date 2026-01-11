#!/usr/bin/env python3
"""Compare Qwen3-Coder vs GPT-OSS-20B on code completion tasks.

Tests both models with LSP context to compare their effectiveness.
"""

import sys
from pathlib import Path

# Test cases with expected API names from kiri project
TEST_CASES = [
    {
        "name": "DuckDB query method",
        "context": """// File: tests/indexer/metadata.spec.ts
// Available methods: all, run, prepare
// Available types: DuckDBClient

import { DuckDBClient } from "../../src/shared/duckdb.js";

async function query(db: DuckDBClient): Promise<void> {
    const result = await db.""",
        "expected_keywords": ["all", "run"],
    },
    {
        "name": "Vitest matchers",
        "context": """// File: tests/server/prompts.spec.ts
// Available methods: describe, expect, it, toBe, toEqual

import { describe, expect, it } from "vitest";

describe("prompts", () => {
    it("should match", async () => {
        expect(actual).to""",
        "expected_keywords": ["toBe", "toEqual", "to"],
    },
    {
        "name": "Kiri runIndexer",
        "context": """// File: src/indexer/cli.ts
// Available methods: runIndexer
// Available types: IndexerOptions

import { runIndexer } from "./index.js";

async function main(): Promise<void> {
    await run""",
        "expected_keywords": ["runIndexer", "Indexer"],
    },
    {
        "name": "Rust dependency",
        "context": """// File: src/indexer/codeintel/rust/analyzer.ts
// Available methods: collectRustDependencies, analyzeRustCode

export async function getDependencies(code: string): Promise<string[]> {
    return await collectRust""",
        "expected_keywords": ["Rust", "Dependencies", "collect"],
    },
]


def test_model(model_path: str, adapter_path: Path, model_name: str):
    """Test a single model on all test cases."""
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"{'='*70}")

    sampler = make_sampler(temp=0.1)

    print(f"Loading {model_name}...")
    model, tokenizer = load(model_path, adapter_path=str(adapter_path))
    print("Model loaded.\n")

    results = []

    for test in TEST_CASES:
        prompt = f"""### Instruction:
Complete the following typescript code:

### Input:
{test['context']}

### Response:
"""
        response = generate(model, tokenizer, prompt=prompt, max_tokens=50, sampler=sampler)

        found = [k for k in test['expected_keywords'] if k in response]
        accuracy = len(found) / len(test['expected_keywords']) * 100

        results.append({
            'name': test['name'],
            'accuracy': accuracy,
            'found': found,
            'expected': test['expected_keywords'],
            'response': response[:80],
        })

        print(f"  {test['name']}: {accuracy:.0f}% ({len(found)}/{len(test['expected_keywords'])})")
        print(f"    -> {response[:60]}...")

    avg = sum(r['accuracy'] for r in results) / len(results)
    print(f"\n  Average: {avg:.1f}%")

    return results, avg


def main():
    print("=" * 70)
    print("Model Comparison: Qwen3-Coder vs GPT-OSS-20B")
    print("=" * 70)

    project_root = Path(__file__).parent.parent

    # Model configurations
    models = {
        "Qwen3-Coder-30B": {
            "path": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
            "adapter": project_root / "output/mlx-qwen3-coder/adapter",
        },
        "GPT-OSS-20B": {
            "path": "lmstudio-community/gpt-oss-20b-MLX-8bit",
            "adapter": project_root / "output/gptoss-20b-lsp/adapter",
        },
    }

    # Check adapters exist
    for name, config in models.items():
        if not config["adapter"].exists():
            print(f"ERROR: Adapter not found for {name}: {config['adapter']}")
            return 1

    try:
        from mlx_lm import generate, load
    except ImportError:
        print("ERROR: mlx-lm not installed")
        return 1

    # Test each model
    all_results = {}
    for name, config in models.items():
        results, avg = test_model(config["path"], config["adapter"], name)
        all_results[name] = {"results": results, "average": avg}

    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n{'Test Case':<25} {'Qwen3-Coder':>15} {'GPT-OSS-20B':>15}")
    print("-" * 55)

    for i, test in enumerate(TEST_CASES):
        qwen_acc = all_results["Qwen3-Coder-30B"]["results"][i]["accuracy"]
        gptoss_acc = all_results["GPT-OSS-20B"]["results"][i]["accuracy"]
        winner = "←" if qwen_acc > gptoss_acc else ("→" if gptoss_acc > qwen_acc else "=")
        print(f"{test['name']:<25} {qwen_acc:>14.0f}% {gptoss_acc:>14.0f}%  {winner}")

    print("-" * 55)
    qwen_avg = all_results["Qwen3-Coder-30B"]["average"]
    gptoss_avg = all_results["GPT-OSS-20B"]["average"]
    winner = "Qwen3-Coder" if qwen_avg > gptoss_avg else ("GPT-OSS-20B" if gptoss_avg > qwen_avg else "Tie")
    print(f"{'AVERAGE':<25} {qwen_avg:>14.1f}% {gptoss_avg:>14.1f}%")
    print(f"\nWinner: {winner}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
