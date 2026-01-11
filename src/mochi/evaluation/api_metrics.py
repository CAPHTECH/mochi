"""API accuracy metrics and golden dataset evaluation.

Evaluates model outputs against golden dataset test cases to measure
API accuracy, hallucination rate, and context compliance.

Law compliance:
- L-eval-accuracy: Measure API usage accuracy
- L-eval-hallucination: Detect and quantify hallucinations
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml


@dataclass
class TestCase:
    """A single test case from the golden dataset."""

    id: str
    category: str
    name: str
    context: str
    input: str
    expected_keywords: list[str]
    forbidden_keywords: list[str] = field(default_factory=list)
    expected_pattern: str | None = None
    instruction: str | None = None
    description: str = ""


@dataclass
class TestResult:
    """Result of evaluating a single test case."""

    test_case_id: str
    passed: bool
    output: str
    # Keyword analysis
    found_expected: list[str] = field(default_factory=list)
    missing_expected: list[str] = field(default_factory=list)
    found_forbidden: list[str] = field(default_factory=list)
    # Pattern analysis
    pattern_matched: bool = False
    # API analysis
    methods_used: list[str] = field(default_factory=list)
    hallucinated_methods: list[str] = field(default_factory=list)
    # Timing
    inference_time_ms: float = 0.0


@dataclass
class APIAccuracyMetrics:
    """Aggregate metrics from golden dataset evaluation.

    Attributes:
        api_precision: Ratio of correct API calls to total API calls
        hallucination_rate: Ratio of hallucinated APIs to total
        forbidden_violation_rate: Rate of forbidden keyword violations
        pattern_match_rate: Rate of expected pattern matches
        context_compliance: Rate of using only context-provided methods
        total_tests: Number of test cases evaluated
        passed_tests: Number of test cases passed
    """

    api_precision: float = 0.0
    hallucination_rate: float = 0.0
    forbidden_violation_rate: float = 0.0
    pattern_match_rate: float = 0.0
    context_compliance: float = 0.0
    total_tests: int = 0
    passed_tests: int = 0
    total_keywords_expected: int = 0
    total_keywords_found: int = 0
    total_forbidden_violations: int = 0
    total_patterns_tested: int = 0
    total_patterns_matched: int = 0

    @property
    def pass_rate(self) -> float:
        """Overall test pass rate."""
        return self.passed_tests / self.total_tests if self.total_tests > 0 else 0.0

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=== API Accuracy Metrics ===",
            f"Pass Rate: {self.pass_rate:.1%} ({self.passed_tests}/{self.total_tests})",
            f"API Precision: {self.api_precision:.1%}",
            f"Hallucination Rate: {self.hallucination_rate:.1%}",
            f"Forbidden Violations: {self.forbidden_violation_rate:.1%}",
            f"Pattern Match Rate: {self.pattern_match_rate:.1%}",
            f"Context Compliance: {self.context_compliance:.1%}",
        ]
        return "\n".join(lines)


class GoldenDatasetEvaluator:
    """Evaluate model outputs against golden dataset.

    Loads test cases from golden_dataset.yaml and evaluates model
    outputs for API accuracy, hallucinations, and pattern compliance.

    Usage:
        evaluator = GoldenDatasetEvaluator("data/golden_dataset.yaml")
        result = evaluator.evaluate_single(test_case_id, model_output)
        metrics = evaluator.evaluate_all(generate_fn)
    """

    def __init__(self, golden_path: str | Path) -> None:
        """Initialize evaluator with golden dataset.

        Args:
            golden_path: Path to golden_dataset.yaml
        """
        self.golden_path = Path(golden_path)
        self.test_cases: dict[str, TestCase] = {}
        self.categories: dict[str, str] = {}
        self.thresholds: dict[str, float] = {}
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load and parse the golden dataset."""
        with open(self.golden_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Load categories
        for cat in data.get("categories", []):
            self.categories[cat["id"]] = cat["name"]

        # Load test cases
        for tc in data.get("test_cases", []):
            test_case = TestCase(
                id=tc["id"],
                category=tc["category"],
                name=tc["name"],
                context=tc.get("context", ""),
                input=tc.get("input", ""),
                expected_keywords=tc.get("expected_keywords", []),
                forbidden_keywords=tc.get("forbidden_keywords", []),
                expected_pattern=tc.get("expected_pattern"),
                instruction=tc.get("instruction"),
                description=tc.get("description", ""),
            )
            self.test_cases[tc["id"]] = test_case

        # Load thresholds
        metrics_config = data.get("metrics", {})
        for metric_name, config in metrics_config.items():
            if isinstance(config, dict) and "threshold" in config:
                self.thresholds[metric_name] = config["threshold"]

    def get_test_cases(self, category: str | None = None) -> list[TestCase]:
        """Get test cases, optionally filtered by category.

        Args:
            category: Optional category filter

        Returns:
            List of matching test cases
        """
        if category:
            return [tc for tc in self.test_cases.values() if tc.category == category]
        return list(self.test_cases.values())

    def evaluate_single(
        self,
        test_case: TestCase | str,
        output: str,
        inference_time_ms: float = 0.0,
    ) -> TestResult:
        """Evaluate a single test case.

        Args:
            test_case: TestCase or test case ID
            output: Model output to evaluate
            inference_time_ms: Inference time for this output

        Returns:
            TestResult with detailed analysis
        """
        if isinstance(test_case, str):
            test_case = self.test_cases[test_case]

        # Check expected keywords
        found_expected = []
        missing_expected = []
        for keyword in test_case.expected_keywords:
            if keyword.lower() in output.lower():
                found_expected.append(keyword)
            else:
                missing_expected.append(keyword)

        # Check forbidden keywords
        found_forbidden = []
        for keyword in test_case.forbidden_keywords:
            if keyword.lower() in output.lower():
                found_forbidden.append(keyword)

        # Check expected pattern
        pattern_matched = False
        if test_case.expected_pattern:
            pattern_matched = bool(re.search(test_case.expected_pattern, output))

        # Extract method calls from output
        methods_used = self._extract_method_calls(output)

        # Check for hallucinated methods (not in context)
        context_methods = self._parse_context_methods(test_case.context)
        hallucinated = [m for m in methods_used if m not in context_methods]

        # Determine if test passed
        passed = (
            len(missing_expected) == 0
            and len(found_forbidden) == 0
            and (not test_case.expected_pattern or pattern_matched)
        )

        return TestResult(
            test_case_id=test_case.id,
            passed=passed,
            output=output,
            found_expected=found_expected,
            missing_expected=missing_expected,
            found_forbidden=found_forbidden,
            pattern_matched=pattern_matched,
            methods_used=methods_used,
            hallucinated_methods=hallucinated,
            inference_time_ms=inference_time_ms,
        )

    def evaluate_all(
        self,
        generate_fn: Callable[[str, str, str], tuple[str, float]],
        category: str | None = None,
    ) -> tuple[APIAccuracyMetrics, list[TestResult]]:
        """Evaluate all test cases using the provided generate function.

        Args:
            generate_fn: Function that takes (instruction, input, context) and
                        returns (output, inference_time_ms)
            category: Optional category filter

        Returns:
            Tuple of (aggregate metrics, list of results)
        """
        test_cases = self.get_test_cases(category)
        results = []

        for tc in test_cases:
            instruction = tc.instruction or f"Complete: {tc.name}"
            full_input = f"{tc.context}\n{tc.input}" if tc.context else tc.input

            try:
                output, time_ms = generate_fn(instruction, full_input, tc.context)
            except Exception as e:
                output = f"Error: {e}"
                time_ms = 0.0

            result = self.evaluate_single(tc, output, time_ms)
            results.append(result)

        # Calculate aggregate metrics
        metrics = self._calculate_metrics(results)

        return metrics, results

    def _calculate_metrics(self, results: list[TestResult]) -> APIAccuracyMetrics:
        """Calculate aggregate metrics from results."""
        if not results:
            return APIAccuracyMetrics()

        total = len(results)
        passed = sum(1 for r in results if r.passed)

        # Keyword metrics
        total_expected = sum(len(r.found_expected) + len(r.missing_expected) for r in results)
        total_found = sum(len(r.found_expected) for r in results)
        total_forbidden = sum(len(r.found_forbidden) for r in results)

        # Pattern metrics
        total_patterns = sum(1 for r in results if r.pattern_matched is not None)
        patterns_matched = sum(1 for r in results if r.pattern_matched)

        # API metrics
        total_methods = sum(len(r.methods_used) for r in results)
        total_hallucinated = sum(len(r.hallucinated_methods) for r in results)

        return APIAccuracyMetrics(
            api_precision=total_found / total_expected if total_expected > 0 else 0.0,
            hallucination_rate=total_hallucinated / total_methods if total_methods > 0 else 0.0,
            forbidden_violation_rate=total_forbidden / total if total > 0 else 0.0,
            pattern_match_rate=patterns_matched / total_patterns if total_patterns > 0 else 0.0,
            context_compliance=1.0 - (total_hallucinated / total_methods if total_methods > 0 else 0.0),
            total_tests=total,
            passed_tests=passed,
            total_keywords_expected=total_expected,
            total_keywords_found=total_found,
            total_forbidden_violations=total_forbidden,
            total_patterns_tested=total_patterns,
            total_patterns_matched=patterns_matched,
        )

    def _extract_method_calls(self, code: str) -> list[str]:
        """Extract method call names from code."""
        methods = []
        pattern = r"\.([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\(|<)"
        for match in re.finditer(pattern, code):
            methods.append(match.group(1))
        return methods

    def _parse_context_methods(self, context: str) -> set[str]:
        """Parse method names from context string."""
        if not context:
            return set()

        methods = set()

        # Pattern: "//   methodName(" or "//   methodName<T>("
        pattern = r"//\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[<(]"
        for match in re.finditer(pattern, context):
            methods.add(match.group(1))

        return methods

    def compare_with_context(
        self,
        output: str,
        context: str,
    ) -> dict[str, Any]:
        """Compare output against context methods.

        Args:
            output: Generated code
            context: LSP context with available methods

        Returns:
            Dictionary with comparison results
        """
        context_methods = self._parse_context_methods(context)
        used_methods = set(self._extract_method_calls(output))

        correct = used_methods & context_methods
        hallucinated = used_methods - context_methods
        unused = context_methods - used_methods

        return {
            "context_methods": list(context_methods),
            "used_methods": list(used_methods),
            "correct_methods": list(correct),
            "hallucinated_methods": list(hallucinated),
            "unused_methods": list(unused),
            "precision": len(correct) / len(used_methods) if used_methods else 1.0,
            "recall": len(correct) / len(context_methods) if context_methods else 1.0,
            "hallucination_rate": len(hallucinated) / len(used_methods) if used_methods else 0.0,
        }


def run_evaluation(
    golden_path: str = "data/golden_dataset.yaml",
    engine: Any = None,
) -> APIAccuracyMetrics:
    """Run full evaluation on golden dataset.

    Args:
        golden_path: Path to golden dataset YAML
        engine: Inference engine with generate() method

    Returns:
        APIAccuracyMetrics with evaluation results
    """
    evaluator = GoldenDatasetEvaluator(golden_path)

    def generate_fn(instruction: str, input_text: str, context: str) -> tuple[str, float]:
        if engine is None:
            # Dry run mode
            return "", 0.0

        result = engine.generate(
            instruction=instruction,
            input_text=input_text,
            context=context,
        )
        return result.response, result.inference_time_ms

    metrics, results = evaluator.evaluate_all(generate_fn)

    print(metrics.summary())
    print(f"\nThreshold Checks:")
    for name, threshold in evaluator.thresholds.items():
        metric_value = getattr(metrics, name, None)
        if metric_value is not None:
            status = "PASS" if metric_value >= threshold else "FAIL"
            print(f"  {name}: {metric_value:.1%} (threshold: {threshold:.1%}) [{status}]")

    return metrics
