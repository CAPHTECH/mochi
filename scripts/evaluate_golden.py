#!/usr/bin/env python3
"""Evaluate model against golden dataset for API accuracy metrics."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mochi.evaluation.api_metrics import GoldenDatasetEvaluator
from mochi.mcp.inference_mlx import MLXInferenceEngine, TaskType, InferenceConfig


def main():
    print("=" * 70)
    print("Golden Dataset Evaluation")
    print("=" * 70)
    print()

    # Load evaluator
    golden_path = Path(__file__).parent.parent / "data" / "golden_dataset.yaml"
    evaluator = GoldenDatasetEvaluator(str(golden_path))

    print(f"Loaded {len(evaluator.test_cases)} test cases")
    print()

    # Load model
    print("Loading model...")
    engine = MLXInferenceEngine(preset="qwen3-coder")
    engine.load()
    print("Model loaded.")
    print()

    # Define generate function
    def generate_fn(instruction: str, input_text: str, context: str) -> tuple[str, float]:
        config = InferenceConfig.for_task(TaskType.METHOD_CALL)
        result = engine.generate_with_config(
            instruction=instruction,
            input_text=input_text,
            context=context,
            task_type=TaskType.METHOD_CALL,
            config=config,
        )
        return result.response, result.inference_time_ms

    # Run evaluation
    print("Running evaluation...")
    print()

    metrics, results = evaluator.evaluate_all(generate_fn)

    # Print results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    print(metrics.summary())
    print()

    # Check thresholds
    print("Threshold Checks:")
    threshold_results = {}
    for name, threshold in evaluator.thresholds.items():
        metric_value = getattr(metrics, name, None)
        if metric_value is not None:
            # For hallucination and violation rates, lower is better
            if "rate" in name and "match" not in name:
                passed = metric_value <= threshold
            else:
                passed = metric_value >= threshold
            status = "PASS" if passed else "FAIL"
            threshold_results[name] = passed
            print(f"  {name}: {metric_value:.1%} (threshold: {threshold:.1%}) [{status}]")
    print()

    # Show individual results
    print("Individual Test Results:")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.test_case_id}")
        if result.missing_expected:
            print(f"        Missing: {result.missing_expected}")
        if result.found_forbidden:
            print(f"        Forbidden: {result.found_forbidden}")
        if result.hallucinated_methods:
            print(f"        Hallucinated: {result.hallucinated_methods}")
    print()

    # Cleanup
    engine.unload()

    # Overall result
    all_passed = all(threshold_results.values()) if threshold_results else False
    if all_passed:
        print("Result: All thresholds met!")
        return 0
    else:
        failed = [k for k, v in threshold_results.items() if not v]
        print(f"Result: Some thresholds not met: {failed}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
