#!/usr/bin/env python3
"""Verify accuracy of GPT-OSS-20B trained model.

Tests the retrained model to ensure LSP context improves API/schema name accuracy.
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
        "description": "Should suggest DuckDB query methods (all, run)"
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
        "description": "Should suggest vitest matchers"
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
        "description": "Should suggest kiri runIndexer function"
    },
    {
        "name": "Rust dependency analysis",
        "context": """// File: src/indexer/codeintel/rust/analyzer.ts
// Available methods: collectRustDependencies, analyzeRustCode

export async function getDependencies(code: string): Promise<string[]> {
    return await collectRust""",
        "expected_keywords": ["Rust", "Dependencies", "collect"],
        "description": "Should suggest collectRustDependencies"
    },
]


def main():
    """Run accuracy verification."""
    print("=" * 70)
    print("GPT-OSS-20B Accuracy Verification: Schema/API Name Generation")
    print("=" * 70)
    print()

    project_root = Path(__file__).parent.parent

    # Check if adapter exists
    adapter_path = project_root / "output" / "gptoss-20b-lsp" / "adapter"
    if not adapter_path.exists():
        print(f"ERROR: Adapter not found at {adapter_path}")
        print("Run scripts/train_gptoss_lora.py first")
        return 1

    print(f"Adapter: {adapter_path}")
    print()

    # Try to load model with mlx_lm
    try:
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        print("ERROR: mlx-lm not installed. Install with: pip install mlx-lm")
        return 1

    # Create sampler with low temperature for deterministic output
    sampler = make_sampler(temp=0.1)

    print("Loading GPT-OSS-20B with adapter...")
    model, tokenizer = load(
        "lmstudio-community/gpt-oss-20b-MLX-8bit",
        adapter_path=str(adapter_path),
    )
    print("Model loaded.\n")

    # Run test cases
    results = []

    for i, test in enumerate(TEST_CASES, 1):
        print(f"--- Test {i}: {test['name']} ---")
        print(f"Description: {test['description']}")
        print()

        # Format as Alpaca prompt
        prompt = f"""### Instruction:
Complete the following typescript code:

### Input:
{test['context']}

### Response:
"""

        # Generate completion
        print("Generating completion...")
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=50,
            sampler=sampler,
        )

        print(f"Generated: {response[:100]}...")
        print()

        # Check if expected keywords appear
        found_keywords = []
        for keyword in test['expected_keywords']:
            if keyword in response:
                found_keywords.append(keyword)

        accuracy = len(found_keywords) / len(test['expected_keywords']) * 100
        results.append({
            'name': test['name'],
            'accuracy': accuracy,
            'found': found_keywords,
            'expected': test['expected_keywords'],
        })

        print(f"Expected keywords: {test['expected_keywords']}")
        print(f"Found keywords: {found_keywords}")
        print(f"Accuracy: {accuracy:.1f}%")
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_accuracy = sum(r['accuracy'] for r in results) / len(results)

    for r in results:
        status = "PASS" if r['accuracy'] >= 50 else "FAIL"
        print(f"  [{status}] {r['name']}: {r['accuracy']:.1f}%")

    print()
    print(f"Overall Accuracy: {total_accuracy:.1f}%")
    print()

    if total_accuracy >= 50:
        print("Result: GPT-OSS-20B with LSP context is performing well!")
        return 0
    else:
        print("Result: Further tuning may be needed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
