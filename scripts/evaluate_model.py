#!/usr/bin/env python3
"""Comprehensive evaluation pipeline for Mochi models.

Evaluates trained models across different task types with structured reporting.
Supports multiple models and produces JSON reports for CI integration.

Usage:
    # Evaluate default model (qwen3-coder)
    python scripts/evaluate_model.py

    # Evaluate specific preset
    python scripts/evaluate_model.py --preset gpt-oss

    # Output JSON report
    python scripts/evaluate_model.py --output-json results.json

    # Run specific task type
    python scripts/evaluate_model.py --task-type method_call
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class TestCase:
    """Individual test case for evaluation."""

    name: str
    task_type: str  # completion, method_call, explanation, documentation, import
    instruction: str
    input_text: str
    context: str
    expected_keywords: list[str]
    description: str
    weight: float = 1.0  # Weight for overall score calculation


@dataclass
class TestResult:
    """Result of a single test case."""

    test_name: str
    task_type: str
    passed: bool
    accuracy: float
    found_keywords: list[str]
    expected_keywords: list[str]
    response: str
    inference_time_ms: float
    tokens_generated: int


@dataclass
class EvaluationReport:
    """Complete evaluation report."""

    model_preset: str
    adapter_path: str
    timestamp: str
    total_tests: int
    passed_tests: int
    overall_accuracy: float
    by_task_type: dict[str, dict[str, Any]] = field(default_factory=dict)
    results: list[TestResult] = field(default_factory=list)
    total_time_seconds: float = 0.0


# Test cases organized by task type
TEST_CASES: list[TestCase] = [
    # === METHOD_CALL tests ===
    TestCase(
        name="DuckDB query method",
        task_type="method_call",
        instruction="Complete the method call:",
        input_text="const result = await db.",
        context="// Available methods:\n//   all(sql: string): Promise<T[]>\n//   run(sql: string): Promise<void>",
        expected_keywords=["all", "run"],
        description="Should suggest DuckDB query methods",
    ),
    TestCase(
        name="Vitest matchers",
        task_type="method_call",
        instruction="Complete the method call:",
        input_text="expect(actual).to",
        context="// Available methods:\n//   toBe(expected: T): void\n//   toEqual(expected: T): void\n//   toContain(item: T): void",
        expected_keywords=["toBe", "toEqual", "to"],
        description="Should suggest vitest matchers",
    ),
    TestCase(
        name="Array methods",
        task_type="method_call",
        instruction="Complete the method call:",
        input_text="const filtered = items.fil",
        context="// Available methods:\n//   filter(fn: (item: T) => boolean): T[]\n//   map(fn: (item: T) => U): U[]\n//   reduce(fn: (acc: U, item: T) => U, init: U): U",
        expected_keywords=["filter", "fil"],
        description="Should suggest array filter method",
    ),
    TestCase(
        name="Promise methods",
        task_type="method_call",
        instruction="Complete the method call:",
        input_text="const results = await Promise.al",
        context="// Available methods:\n//   all(promises: Promise<T>[]): Promise<T[]>\n//   race(promises: Promise<T>[]): Promise<T>\n//   allSettled(promises: Promise<T>[]): Promise<PromiseSettledResult<T>[]>",
        expected_keywords=["all", "allSettled"],
        description="Should suggest Promise.all or allSettled",
    ),
    # === COMPLETION tests ===
    TestCase(
        name="Function completion",
        task_type="completion",
        instruction="Complete the following code:",
        input_text="async function fetchUser(id: number): Promise<User> {\n    return await db.",
        context="// Types: User { id: number; name: string; email: string }\n// Available methods:\n//   all(sql: string, params?: any[]): Promise<T[]>",
        expected_keywords=["all", "SELECT", "User", "id"],
        description="Should complete database query",
    ),
    TestCase(
        name="Class method completion",
        task_type="completion",
        instruction="Complete the following code:",
        input_text="class UserService {\n    constructor(private db: Database) {}\n    \n    async getById(id: number): Promise<User | null> {\n        const users = await this.db.",
        context="// Types: User, Database\n// Available methods:\n//   all(sql: string): Promise<T[]>\n//   run(sql: string): Promise<void>",
        expected_keywords=["all", "db"],
        description="Should complete class method with db call",
    ),
    # === IMPORT tests ===
    TestCase(
        name="Import completion",
        task_type="import",
        instruction="Add the necessary import statements:",
        input_text="export async function handle(req: Request): Promise<Response> {\n    const db = new DuckDBClient();\n    const users = await db.all(\"SELECT * FROM users\");\n    return new Response(JSON.stringify(users));\n}",
        context="// Available modules:\n//   ./shared/duckdb.js: DuckDBClient\n//   ./types.js: User, Response",
        expected_keywords=["import", "DuckDBClient", "duckdb"],
        description="Should suggest necessary imports",
    ),
    # === DOCUMENTATION tests ===
    TestCase(
        name="Function documentation",
        task_type="documentation",
        instruction="Add documentation comments to this code:",
        input_text="async function fetchUsers(limit: number = 10): Promise<User[]> {\n    return await db.all(`SELECT * FROM users LIMIT ${limit}`);\n}",
        context="",
        expected_keywords=["@param", "@returns", "Promise", "User"],
        description="Should generate JSDoc documentation",
    ),
    # === EXPLANATION tests ===
    TestCase(
        name="Code explanation",
        task_type="explanation",
        instruction="Explain what this code does:",
        input_text="const memoize = <T, U>(fn: (arg: T) => U): (arg: T) => U => {\n    const cache = new Map<T, U>();\n    return (arg: T) => {\n        if (cache.has(arg)) return cache.get(arg)!;\n        const result = fn(arg);\n        cache.set(arg, result);\n        return result;\n    };\n};",
        context="",
        expected_keywords=["cache", "memoize", "function", "result"],
        description="Should explain memoization function",
    ),
]


def evaluate_model(
    preset: str = "qwen3-coder",
    task_type_filter: Optional[str] = None,
    verbose: bool = True,
) -> EvaluationReport:
    """Run evaluation on the model.

    Args:
        preset: Model preset ("qwen3-coder" or "gpt-oss")
        task_type_filter: Only run tests for specific task type
        verbose: Print detailed output

    Returns:
        EvaluationReport with all results
    """
    from mochi.mcp.inference_mlx import (
        MLXInferenceEngine,
        TaskType,
        InferenceConfig,
    )

    # Initialize engine
    engine = MLXInferenceEngine(preset=preset)

    if verbose:
        print("=" * 70)
        print(f"Mochi Model Evaluation: {preset}")
        print("=" * 70)
        print()
        print(f"Model: {engine.model_path}")
        print(f"Adapter: {engine.adapter_path}")
        print()

    # Check adapter exists
    if engine.adapter_path and not engine.adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {engine.adapter_path}")

    # Load model
    if verbose:
        print("Loading model...")
    engine.load()
    if verbose:
        print("Model loaded.\n")

    # Filter test cases
    test_cases = TEST_CASES
    if task_type_filter:
        test_cases = [t for t in TEST_CASES if t.task_type == task_type_filter]

    # Run tests
    results: list[TestResult] = []
    start_time = time.time()

    for i, test in enumerate(test_cases, 1):
        if verbose:
            print(f"--- Test {i}/{len(test_cases)}: {test.name} ---")
            print(f"Task Type: {test.task_type}")
            print(f"Description: {test.description}")
            print()

        # Map task type string to enum
        task_type_enum = TaskType(test.task_type)

        # Use task-specific config
        config = InferenceConfig.for_task(task_type_enum)

        # Generate
        result = engine.generate_with_config(
            instruction=test.instruction,
            input_text=test.input_text,
            context=test.context,
            task_type=task_type_enum,
            config=config,
        )

        # Check keywords
        found_keywords = [
            kw for kw in test.expected_keywords if kw.lower() in result.response.lower()
        ]
        accuracy = (
            len(found_keywords) / len(test.expected_keywords) * 100
            if test.expected_keywords
            else 100.0
        )
        passed = accuracy >= 50

        test_result = TestResult(
            test_name=test.name,
            task_type=test.task_type,
            passed=passed,
            accuracy=accuracy,
            found_keywords=found_keywords,
            expected_keywords=test.expected_keywords,
            response=result.response[:200],  # Truncate for report
            inference_time_ms=result.inference_time_ms,
            tokens_generated=result.tokens_generated,
        )
        results.append(test_result)

        if verbose:
            print(f"Response: {result.response[:100]}...")
            print(f"Expected: {test.expected_keywords}")
            print(f"Found: {found_keywords}")
            print(f"Accuracy: {accuracy:.1f}% {'[PASS]' if passed else '[FAIL]'}")
            print(f"Time: {result.inference_time_ms:.0f}ms")
            print()

    total_time = time.time() - start_time

    # Calculate metrics by task type
    by_task_type: dict[str, dict[str, Any]] = {}
    for task_type in set(r.task_type for r in results):
        task_results = [r for r in results if r.task_type == task_type]
        by_task_type[task_type] = {
            "total": len(task_results),
            "passed": sum(1 for r in task_results if r.passed),
            "accuracy": sum(r.accuracy for r in task_results) / len(task_results),
            "avg_time_ms": sum(r.inference_time_ms for r in task_results)
            / len(task_results),
        }

    # Build report
    report = EvaluationReport(
        model_preset=preset,
        adapter_path=str(engine.adapter_path),
        timestamp=datetime.now().isoformat(),
        total_tests=len(results),
        passed_tests=sum(1 for r in results if r.passed),
        overall_accuracy=sum(r.accuracy for r in results) / len(results),
        by_task_type=by_task_type,
        results=results,
        total_time_seconds=total_time,
    )

    # Cleanup
    engine.unload()

    return report


def print_summary(report: EvaluationReport) -> None:
    """Print evaluation summary."""
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print()
    print(f"Model: {report.model_preset}")
    print(f"Adapter: {report.adapter_path}")
    print(f"Timestamp: {report.timestamp}")
    print()

    print("Results by Task Type:")
    for task_type, metrics in report.by_task_type.items():
        status = "PASS" if metrics["accuracy"] >= 50 else "FAIL"
        print(
            f"  [{status}] {task_type}: "
            f"{metrics['passed']}/{metrics['total']} tests, "
            f"{metrics['accuracy']:.1f}% accuracy, "
            f"{metrics['avg_time_ms']:.0f}ms avg"
        )
    print()

    print("Individual Results:")
    for r in report.results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.test_name}: {r.accuracy:.1f}%")
    print()

    print(f"Overall: {report.passed_tests}/{report.total_tests} tests passed")
    print(f"Overall Accuracy: {report.overall_accuracy:.1f}%")
    print(f"Total Time: {report.total_time_seconds:.1f}s")
    print()

    if report.overall_accuracy >= 50:
        print("Result: Model meets accuracy threshold (>= 50%)")
    else:
        print("Result: Model needs improvement (< 50%)")


def save_json_report(report: EvaluationReport, output_path: str) -> None:
    """Save report as JSON file."""
    # Convert dataclasses to dicts
    report_dict = {
        "model_preset": report.model_preset,
        "adapter_path": report.adapter_path,
        "timestamp": report.timestamp,
        "total_tests": report.total_tests,
        "passed_tests": report.passed_tests,
        "overall_accuracy": report.overall_accuracy,
        "by_task_type": report.by_task_type,
        "total_time_seconds": report.total_time_seconds,
        "results": [asdict(r) for r in report.results],
    }

    with open(output_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    print(f"Report saved to: {output_path}")


def main() -> int:
    """Run evaluation with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Mochi model accuracy across task types"
    )
    parser.add_argument(
        "--preset",
        choices=["qwen3-coder", "gpt-oss"],
        default="qwen3-coder",
        help="Model preset to evaluate",
    )
    parser.add_argument(
        "--task-type",
        choices=["completion", "method_call", "explanation", "documentation", "import"],
        help="Only run tests for specific task type",
    )
    parser.add_argument(
        "--output-json",
        metavar="PATH",
        help="Save JSON report to file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output (only summary)",
    )

    args = parser.parse_args()

    try:
        report = evaluate_model(
            preset=args.preset,
            task_type_filter=args.task_type,
            verbose=not args.quiet,
        )

        print_summary(report)

        if args.output_json:
            save_json_report(report, args.output_json)

        return 0 if report.overall_accuracy >= 50 else 1

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
