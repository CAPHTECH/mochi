"""Classify and validate code transformation patterns.

Provides both rule-based classification (fast, no external dependencies)
and optional LLM-assisted classification (more accurate, requires API).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

from .diff_extractor import CodeTransformPair

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of transformation classification."""

    transform_type: str
    is_learnable: bool
    instruction: str
    confidence: float  # 0.0-1.0
    reason: str


class LLMClient(Protocol):
    """Protocol for LLM client interface."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt."""
        ...


class PatternClassifier:
    """Classify code transformation patterns.

    Supports two modes:
    1. Rule-based: Fast classification using pattern matching
    2. LLM-assisted: More accurate using LLM for validation

    The rule-based classifier is always used first, with LLM as optional
    validation for borderline cases.
    """

    # Instruction templates for each transform type
    INSTRUCTION_TEMPLATES: dict[str, list[str]] = {
        "error-handling": [
            "Add error handling to this function",
            "Wrap this code with try-catch and handle errors appropriately",
            "Add proper error handling following TypeScript best practices",
            "Handle potential errors in this code",
            "Add error recovery to this function",
        ],
        "null-safety": [
            "Add null safety checks to this code",
            "Make this code null-safe using optional chaining",
            "Add proper null checks to prevent runtime errors",
            "Handle null and undefined values safely",
            "Add defensive null handling",
        ],
        "type-safety": [
            "Add type annotations to this code",
            "Improve type safety of this function",
            "Add proper TypeScript types",
            "Add type guards where appropriate",
            "Strengthen the types in this code",
        ],
        "async-await": [
            "Convert this callback-based code to async/await",
            "Refactor to use async/await pattern",
            "Modernize this async code using await",
            "Convert Promise chains to async/await",
        ],
        "validation": [
            "Add input validation to this function",
            "Add runtime validation using zod",
            "Validate the input parameters",
            "Add schema validation to this code",
            "Add assertions to verify input",
        ],
        # Test patterns
        "test-structure": [
            "Write a unit test for this function",
            "Create a test case for this implementation",
            "Complete the test describe/it structure",
            "Write the test body with proper assertions",
            "Add a test case for this behavior",
        ],
        "test-assertion": [
            "Add assertions to verify the behavior",
            "Write expect statements to validate",
            "Add proper assertions for this test",
            "Complete the expect statements",
            "Verify the expected behavior with assertions",
        ],
        "test-setup": [
            "Set up test fixtures",
            "Write the setup/teardown for this test",
            "Initialize test dependencies in beforeEach",
            "Add proper test setup and cleanup",
        ],
        "test-mock": [
            "Set up mocks for this dependency",
            "Create mock implementations",
            "Mock the external dependencies",
            "Write mock setup for this test",
        ],
    }

    # Quality heuristics for each transform type
    QUALITY_PATTERNS: dict[str, dict[str, list[str]]] = {
        "error-handling": {
            "good": [
                r"try\s*\{[\s\S]+catch\s*\(",  # Complete try-catch
                r"throw\s+new\s+\w+Error",  # Typed errors
                r"catch\s*\(\s*\w+\s*:\s*\w+\s*\)",  # Typed catch
            ],
            "bad": [
                r"catch\s*\(\s*\)\s*\{\s*\}",  # Empty catch
                r"catch\s*\(\s*_\s*\)",  # Ignored error
            ],
        },
        "null-safety": {
            "good": [
                r"\?\.\w+",  # Optional chaining
                r"\?\?\s*['\"\w]",  # Nullish coalescing with default
                r"if\s*\(\s*\w+\s*!==?\s*null",  # Explicit null check
            ],
            "bad": [
                r"!\s*\.",  # Non-null assertion (unsafe)
            ],
        },
        "type-safety": {
            "good": [
                r":\s*[A-Z][a-zA-Z0-9_<>]+",  # Type annotation
                r"as\s+const",  # Const assertion
                r"is\s+[A-Z]\w+",  # Type guard
            ],
            "bad": [
                r":\s*any\b",  # any type
                r"as\s+any\b",  # Cast to any
            ],
        },
        "async-await": {
            "good": [
                r"async\s+function\s+\w+",  # Named async function
                r"await\s+\w+\s*\(",  # Await call
            ],
            "bad": [
                r"\.then\s*\(",  # Still using .then()
            ],
        },
        "validation": {
            "good": [
                r"z\.\w+\s*\(\)",  # Zod schema
                r"\.parse\s*\(",  # Schema parsing
                r"assert\w*\s*\(",  # Assertions
            ],
            "bad": [],
        },
        # Test patterns
        "test-structure": {
            "good": [
                r"describe\s*\(['\"]",  # describe block with name
                r"it\s*\(['\"]",  # it block with name
                r"test\s*\(['\"]",  # test block with name
                r"expect\s*\(",  # Has assertions
            ],
            "bad": [
                r"it\.skip\s*\(",  # Skipped test
                r"describe\.skip\s*\(",  # Skipped describe
                r"it\s*\(['\"]['\"]",  # Empty test name
            ],
        },
        "test-assertion": {
            "good": [
                r"expect\s*\([^)]+\)\s*\.\w+",  # Complete expect chain
                r"\.toBe\s*\(",  # toBe assertion
                r"\.toEqual\s*\(",  # toEqual assertion
                r"\.toHaveBeenCalled",  # Mock assertion
                r"\.toThrow\s*\(",  # Error assertion
            ],
            "bad": [
                r"expect\s*\(\s*\)",  # Empty expect
            ],
        },
        "test-setup": {
            "good": [
                r"beforeEach\s*\(\s*(?:async\s*)?\(\s*\)\s*=>",  # Arrow function setup
                r"afterEach\s*\(\s*(?:async\s*)?\(\s*\)\s*=>",  # Arrow function teardown
                r"beforeAll\s*\(",  # beforeAll hook
                r"afterAll\s*\(",  # afterAll hook
            ],
            "bad": [
                r"beforeEach\s*\(\s*\(\s*\)\s*=>\s*\{\s*\}\s*\)",  # Empty beforeEach
            ],
        },
        "test-mock": {
            "good": [
                r"vi\.mock\s*\(['\"]",  # Vitest mock with path
                r"jest\.mock\s*\(['\"]",  # Jest mock with path
                r"vi\.fn\s*\(\s*\)",  # Vitest mock function
                r"jest\.fn\s*\(\s*\)",  # Jest mock function
                r"\.mockResolvedValue\s*\(",  # Mock resolved value
                r"\.mockReturnValue\s*\(",  # Mock return value
            ],
            "bad": [],
        },
    }

    # LLM classification prompt
    CLASSIFICATION_PROMPT = """Analyze this code transformation and classify it.

## Before
```typescript
{before_code}
```

## After
```typescript
{after_code}
```

## Classification Task
1. Transform type: [error-handling|null-safety|type-safety|async-await|validation|other]
2. Is this a clean, learnable example? Consider:
   - Is the transformation focused and complete?
   - Does it follow best practices?
   - Would it teach good patterns to a model?
3. Write an imperative instruction for this transformation (e.g., "Add try-catch error handling")

Respond ONLY in JSON format:
{{
  "type": "...",
  "is_learnable": true/false,
  "instruction": "...",
  "reason": "..."
}}"""

    def __init__(self, llm_client: LLMClient | None = None):
        """Initialize classifier.

        Args:
            llm_client: Optional LLM client for enhanced classification
        """
        self.llm_client = llm_client

    def classify(
        self, pair: CodeTransformPair, use_llm: bool = False
    ) -> ClassificationResult:
        """Classify a transformation pair.

        Args:
            pair: Code transformation pair to classify
            use_llm: Whether to use LLM for validation

        Returns:
            ClassificationResult with type, learnability, and instruction
        """
        # Rule-based classification first
        rule_result = self._classify_rule_based(pair)

        # If LLM is requested and available, validate with LLM
        if use_llm and self.llm_client and rule_result.confidence < 0.9:
            llm_result = self._classify_with_llm(pair)
            if llm_result:
                return llm_result

        return rule_result

    def _classify_rule_based(self, pair: CodeTransformPair) -> ClassificationResult:
        """Classify using rule-based heuristics."""
        transform_type = pair.transform_type  # Already classified by extractor

        # Check quality patterns
        after_code = pair.after_code
        good_patterns = self.QUALITY_PATTERNS.get(transform_type, {}).get("good", [])
        bad_patterns = self.QUALITY_PATTERNS.get(transform_type, {}).get("bad", [])

        good_matches = sum(
            1 for p in good_patterns if re.search(p, after_code, re.MULTILINE)
        )
        bad_matches = sum(
            1 for p in bad_patterns if re.search(p, after_code, re.MULTILINE)
        )

        # Calculate confidence and learnability
        if good_patterns:
            quality_score = good_matches / len(good_patterns)
        else:
            quality_score = 0.5

        is_learnable = good_matches > 0 and bad_matches == 0 and quality_score >= 0.3
        confidence = min(1.0, quality_score + (0.1 if bad_matches == 0 else -0.2))

        # Generate instruction
        templates = self.INSTRUCTION_TEMPLATES.get(transform_type, ["Transform this code"])
        import random

        instruction = random.choice(templates)

        reason = f"Rule-based: {good_matches} good patterns, {bad_matches} bad patterns"

        return ClassificationResult(
            transform_type=transform_type,
            is_learnable=is_learnable,
            instruction=instruction,
            confidence=max(0.0, min(1.0, confidence)),
            reason=reason,
        )

    def _classify_with_llm(self, pair: CodeTransformPair) -> ClassificationResult | None:
        """Classify using LLM for validation."""
        if not self.llm_client:
            return None

        prompt = self.CLASSIFICATION_PROMPT.format(
            before_code=pair.before_code[:1500],  # Truncate for token limit
            after_code=pair.after_code[:1500],
        )

        try:
            response = self.llm_client.generate(prompt, temperature=0.1)
            result = self._parse_llm_response(response)

            if result:
                return ClassificationResult(
                    transform_type=result.get("type", pair.transform_type),
                    is_learnable=result.get("is_learnable", False),
                    instruction=result.get("instruction", "Transform this code"),
                    confidence=0.9 if result.get("is_learnable") else 0.5,
                    reason=f"LLM: {result.get('reason', 'No reason provided')}",
                )
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")

        return None

    def _parse_llm_response(self, response: str) -> dict[str, Any] | None:
        """Parse JSON response from LLM."""
        # Try to extract JSON from response
        try:
            # First try direct parse
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON in response
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return None

    def filter_learnable(
        self, pairs: list[CodeTransformPair], use_llm: bool = False
    ) -> list[tuple[CodeTransformPair, ClassificationResult]]:
        """Filter pairs to only learnable examples.

        Args:
            pairs: List of transformation pairs
            use_llm: Whether to use LLM for validation

        Returns:
            List of (pair, classification) tuples for learnable examples
        """
        learnable: list[tuple[CodeTransformPair, ClassificationResult]] = []

        for pair in pairs:
            result = self.classify(pair, use_llm=use_llm)
            if result.is_learnable:
                learnable.append((pair, result))

        logger.info(f"Filtered to {len(learnable)}/{len(pairs)} learnable examples")
        return learnable
