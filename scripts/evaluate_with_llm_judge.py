#!/usr/bin/env python3
"""Evaluate adapters using LLM-as-Judge (gpt-oss:120b via Ollama).

Uses the same evaluation criteria as src/mochi/assay/llm-judge.ts:
- API Correctness
- Convention Adherence
- Syntax Correctness
- Semantic Appropriateness
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import requests


@dataclass
class JudgeResult:
    """Result from LLM Judge evaluation."""
    score: float
    reasoning: str
    api_correctness: float
    convention_adherence: float
    syntax_correctness: float
    semantic_appropriateness: float
    issues: list[str]


@dataclass
class CombinedResult:
    """Combined evaluation result."""
    keyword_score: float
    judge_result: JudgeResult
    overall_score: float


# Test cases (same as evaluate_comprehensive.py)
TEST_CASES = [
    # Common patterns
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
    # Kiri-specific patterns
    {
        "name": "singleton-registry",
        "category": "kiri",
        "instruction": "Implement a singleton registry pattern for managing language analyzers",
        "input": """interface LanguageAnalyzer {
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
        "name": "config-validation",
        "category": "kiri",
        "instruction": "Add validation for scoring weights configuration with backward compatibility",
        "input": """interface ScoringWeights {
  textMatch: number;
  pathMatch: number;
  graphInbound?: number;
}""",
        "expected_patterns": ["typeof", "undefined", "isFinite", "throw"],
    },
]


class OllamaClient:
    """Simple Ollama client for LLM inference."""

    def __init__(self, endpoint: str = "http://localhost:11434", model: str = "gpt-oss:120b"):
        self.endpoint = endpoint
        self.model = model

    def ping(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = requests.get(f"{self.endpoint}/api/tags", timeout=5)
            return response.ok
        except Exception:
            return False

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512, timeout: int = 300) -> str:
        """Generate text using Ollama API."""
        response = requests.post(
            f"{self.endpoint}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["response"]


class LLMJudge:
    """LLM-as-Judge evaluator using gpt-oss:120b."""

    def __init__(self, ollama: OllamaClient):
        self.ollama = ollama

    def build_prompt(self, instruction: str, input_code: str, output: str, expected_patterns: list[str]) -> str:
        """Build the judge prompt."""
        return f"""You are a code quality evaluator. Evaluate the generated code completion.

## Task
{instruction}

## Code Input
{input_code}

## Generated Output
{output}

## Expected Keywords (should appear)
{", ".join(expected_patterns)}

## Evaluation Criteria
Rate each from 0.0 to 1.0:

1. API_CORRECTNESS: Uses correct methods/APIs, not hallucinated ones
2. CONVENTION_ADHERENCE: Follows TypeScript/JavaScript conventions
3. SYNTAX_CORRECTNESS: Syntactically valid code
4. SEMANTIC_APPROPRIATENESS: Code makes sense for the task and contains expected patterns

## Response Format (REQUIRED - follow exactly)
API_CORRECTNESS: X.X
CONVENTION_ADHERENCE: X.X
SYNTAX_CORRECTNESS: X.X
SEMANTIC_APPROPRIATENESS: X.X
OVERALL: X.X
ISSUES: [comma-separated list or "none"]
REASONING: Brief explanation

Evaluate now:"""

    def parse_response(self, response: str) -> JudgeResult:
        """Parse the judge response."""
        result = JudgeResult(
            score=0.0,
            reasoning="",
            api_correctness=0.0,
            convention_adherence=0.0,
            syntax_correctness=0.0,
            semantic_appropriateness=0.0,
            issues=[],
        )

        for line in response.split("\n"):
            trimmed = line.strip()

            if trimmed.startswith("API_CORRECTNESS:"):
                result.api_correctness = self._parse_score(trimmed)
            elif trimmed.startswith("CONVENTION_ADHERENCE:"):
                result.convention_adherence = self._parse_score(trimmed)
            elif trimmed.startswith("SYNTAX_CORRECTNESS:"):
                result.syntax_correctness = self._parse_score(trimmed)
            elif trimmed.startswith("SEMANTIC_APPROPRIATENESS:"):
                result.semantic_appropriateness = self._parse_score(trimmed)
            elif trimmed.startswith("OVERALL:"):
                result.score = self._parse_score(trimmed)
            elif trimmed.startswith("ISSUES:"):
                issues_str = trimmed[7:].strip()
                if issues_str.lower() != "none" and issues_str != "[]":
                    result.issues = [s.strip() for s in issues_str.strip("[]").split(",") if s.strip()]
            elif trimmed.startswith("REASONING:"):
                result.reasoning = trimmed[10:].strip()

        # Calculate overall if not provided
        if result.score == 0:
            result.score = (
                result.api_correctness +
                result.convention_adherence +
                result.syntax_correctness +
                result.semantic_appropriateness
            ) / 4

        return result

    def _parse_score(self, line: str) -> float:
        """Parse score from line like 'CRITERIA: 0.8'."""
        match = re.search(r":\s*([\d.]+)", line)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
        return 0.0

    def evaluate(self, instruction: str, input_code: str, output: str, expected_patterns: list[str]) -> JudgeResult:
        """Evaluate code completion using LLM Judge."""
        prompt = self.build_prompt(instruction, input_code, output, expected_patterns)
        response = self.ollama.generate(prompt, temperature=0.0, max_tokens=512, timeout=300)
        return self.parse_response(response)


def combined_evaluate(
    instruction: str,
    input_code: str,
    output: str,
    expected_patterns: list[str],
    judge: LLMJudge,
) -> CombinedResult:
    """Combined evaluation: keyword matching + LLM Judge."""
    # Keyword score
    output_lower = output.lower()
    matches = sum(1 for p in expected_patterns if p.lower() in output_lower)
    keyword_score = matches / len(expected_patterns) if expected_patterns else 1.0

    # LLM Judge
    judge_result = judge.evaluate(instruction, input_code, output, expected_patterns)

    # Combined (30% keyword, 70% judge)
    overall_score = 0.3 * keyword_score + 0.7 * judge_result.score

    return CombinedResult(
        keyword_score=keyword_score,
        judge_result=judge_result,
        overall_score=overall_score,
    )


def generate_completion(model, tokenizer, instruction: str, input_code: str) -> str:
    """Generate completion using MLX model."""
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler

    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_code}\n\n### Response:\n"
    sampler = make_sampler(temp=0.1)

    response = mlx_generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=512,
        sampler=sampler,
    )

    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    return response


def evaluate_adapter(
    model_name: str,
    adapter_path: Path | None,
    test_cases: list[dict],
    label: str,
    judge: LLMJudge,
) -> dict:
    """Evaluate adapter with LLM Judge."""
    from mlx_lm import load

    print(f"\n{'=' * 70}")
    print(f"Evaluating: {label}")
    print(f"{'=' * 70}")

    if adapter_path:
        model, tokenizer = load(model_name, adapter_path=str(adapter_path))
    else:
        model, tokenizer = load(model_name)

    results = []
    total_keyword = 0
    total_judge = 0
    total_overall = 0
    category_scores = {}

    for i, test_case in enumerate(test_cases):
        cat = test_case["category"]
        print(f"\n[{i+1}/{len(test_cases)}] [{cat}] {test_case['name']}")
        sys.stdout.flush()

        # Generate completion
        print("  Generating...", end=" ")
        start_time = time.time()
        output = generate_completion(model, tokenizer, test_case["instruction"], test_case["input"])
        gen_time = time.time() - start_time
        print(f"({gen_time:.1f}s)")

        # Evaluate with LLM Judge
        print("  Judging...", end=" ")
        start_time = time.time()
        result = combined_evaluate(
            test_case["instruction"],
            test_case["input"],
            output,
            test_case["expected_patterns"],
            judge,
        )
        judge_time = time.time() - start_time
        print(f"({judge_time:.1f}s)")

        print(f"  Keyword: {result.keyword_score:.0%}")
        print(f"  Judge: {result.judge_result.score:.0%}")
        print(f"    - API: {result.judge_result.api_correctness:.1f}")
        print(f"    - Convention: {result.judge_result.convention_adherence:.1f}")
        print(f"    - Syntax: {result.judge_result.syntax_correctness:.1f}")
        print(f"    - Semantic: {result.judge_result.semantic_appropriateness:.1f}")
        print(f"  Overall: {result.overall_score:.0%}")
        if result.judge_result.issues:
            print(f"  Issues: {', '.join(result.judge_result.issues)}")
        if result.judge_result.reasoning:
            print(f"  Reasoning: {result.judge_result.reasoning[:100]}...")

        results.append({
            "name": test_case["name"],
            "category": cat,
            "keyword_score": result.keyword_score,
            "judge_score": result.judge_result.score,
            "overall_score": result.overall_score,
            "criteria": {
                "api_correctness": result.judge_result.api_correctness,
                "convention_adherence": result.judge_result.convention_adherence,
                "syntax_correctness": result.judge_result.syntax_correctness,
                "semantic_appropriateness": result.judge_result.semantic_appropriateness,
            },
            "issues": result.judge_result.issues,
            "reasoning": result.judge_result.reasoning,
            "output": output[:500],
        })

        total_keyword += result.keyword_score
        total_judge += result.judge_result.score
        total_overall += result.overall_score

        if cat not in category_scores:
            category_scores[cat] = {"total": 0, "count": 0}
        category_scores[cat]["total"] += result.overall_score
        category_scores[cat]["count"] += 1

    avg_keyword = total_keyword / len(test_cases)
    avg_judge = total_judge / len(test_cases)
    avg_overall = total_overall / len(test_cases)
    category_avgs = {cat: data["total"] / data["count"] for cat, data in category_scores.items()}

    print(f"\n--- Summary: {label} ---")
    print(f"Keyword Score: {avg_keyword:.1%}")
    print(f"Judge Score: {avg_judge:.1%}")
    print(f"Overall Score: {avg_overall:.1%}")
    for cat, score in category_avgs.items():
        print(f"  {cat}: {score:.1%}")

    return {
        "label": label,
        "avg_keyword": avg_keyword,
        "avg_judge": avg_judge,
        "avg_overall": avg_overall,
        "category_scores": category_avgs,
        "results": results,
    }


def main():
    project_root = Path(__file__).parent.parent
    model_name = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"

    # Check Ollama
    ollama = OllamaClient()
    if not ollama.ping():
        print("ERROR: Ollama server not available at http://localhost:11434")
        print("Start Ollama with: ollama serve")
        return 1

    print("=" * 70)
    print("LLM-as-Judge Evaluation (gpt-oss:120b)")
    print("=" * 70)
    print(f"Judge Model: {ollama.model}")
    print(f"Test Cases: {len(TEST_CASES)}")

    judge = LLMJudge(ollama)

    adapters = [
        (project_root / "output" / "base-adapter" / "adapter", "Base Adapter"),
        (project_root / "output" / "kiri-adapter-enhanced" / "adapter", "Enhanced Mixed"),
    ]

    all_results = []

    for adapter_path, label in adapters:
        if adapter_path.exists():
            result = evaluate_adapter(model_name, adapter_path, TEST_CASES, label, judge)
            all_results.append(result)
        else:
            print(f"\nSkipping: {label} (not found)")

    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON (LLM-as-Judge)")
    print("=" * 70)

    print("\n{:<25} {:>10} {:>10} {:>10}".format("Adapter", "Keyword", "Judge", "Overall"))
    print("-" * 58)
    for r in all_results:
        print("{:<25} {:>9.1%} {:>9.1%} {:>9.1%}".format(
            r["label"][:25], r["avg_keyword"], r["avg_judge"], r["avg_overall"]
        ))

    # Save results
    output_file = project_root / "output" / "llm_judge_evaluation.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
