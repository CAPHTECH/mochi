"""Output validation for detecting hallucinated API calls.

Validates generated code against available context (methods, types)
to detect hallucinated API names that don't exist in the codebase.

Law compliance:
- L-hallucination-detect: Identify generated methods not in context
- L-context-validate: Verify generated code uses available APIs
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Result of output validation.

    Attributes:
        is_valid: True if all used methods are in available context
        hallucinated_methods: List of method names not found in context
        used_methods: All method calls found in the output
        available_methods: Methods available from context
        confidence: Confidence score (0-1) of validation
    """

    is_valid: bool
    hallucinated_methods: list[str] = field(default_factory=list)
    used_methods: list[str] = field(default_factory=list)
    available_methods: list[str] = field(default_factory=list)
    confidence: float = 1.0

    @property
    def hallucination_rate(self) -> float:
        """Calculate hallucination rate as ratio of hallucinated to used methods."""
        if not self.used_methods:
            return 0.0
        return len(self.hallucinated_methods) / len(self.used_methods)


class OutputValidator:
    """Validate generated code output against available context.

    Detects hallucinated API calls by comparing method calls in generated
    code against the methods provided in the LSP context.

    Usage:
        validator = OutputValidator()
        result = validator.validate(generated_code, context)
        if not result.is_valid:
            print(f"Hallucinated: {result.hallucinated_methods}")
    """

    # Common method names that are always valid (language builtins)
    ALWAYS_VALID_METHODS: frozenset[str] = frozenset({
        # JavaScript/TypeScript
        "toString", "valueOf", "hasOwnProperty", "isPrototypeOf",
        "propertyIsEnumerable", "toLocaleString", "constructor",
        # Array methods
        "push", "pop", "shift", "unshift", "slice", "splice",
        "concat", "join", "reverse", "sort", "indexOf", "lastIndexOf",
        "includes", "find", "findIndex", "filter", "map", "reduce",
        "reduceRight", "every", "some", "forEach", "flat", "flatMap",
        "fill", "copyWithin", "entries", "keys", "values", "at",
        # String methods
        "charAt", "charCodeAt", "codePointAt", "split", "substring",
        "substr", "toLowerCase", "toUpperCase", "trim", "trimStart",
        "trimEnd", "padStart", "padEnd", "repeat", "replace", "replaceAll",
        "match", "matchAll", "search", "startsWith", "endsWith",
        # Promise methods
        "then", "catch", "finally",
        # Object methods (common ones)
        "assign", "create", "defineProperty", "freeze", "seal",
    })

    def __init__(
        self,
        strict_mode: bool = False,
        ignore_common_methods: bool = True,
    ) -> None:
        """Initialize validator.

        Args:
            strict_mode: If True, require exact method name match
            ignore_common_methods: If True, skip validation for common builtins
        """
        self.strict_mode = strict_mode
        self.ignore_common_methods = ignore_common_methods

    def validate(
        self,
        output: str,
        context: str,
        additional_valid_methods: list[str] | None = None,
    ) -> ValidationResult:
        """Validate generated output against context.

        Args:
            output: Generated code output to validate
            context: LSP context string with available methods
            additional_valid_methods: Extra valid methods to allow

        Returns:
            ValidationResult with hallucination detection
        """
        if not output:
            return ValidationResult(
                is_valid=True,
                confidence=1.0,
            )

        # Parse available methods from context
        available = self._parse_context(context)
        if additional_valid_methods:
            available.update(additional_valid_methods)

        # Add common methods if not in strict mode
        if self.ignore_common_methods:
            available.update(self.ALWAYS_VALID_METHODS)

        # Extract method calls from output
        used = self._extract_method_calls(output)

        # Find hallucinated methods
        hallucinated = [m for m in used if m not in available]

        # Calculate confidence based on context quality
        confidence = 1.0 if available else 0.5  # Lower confidence if no context

        return ValidationResult(
            is_valid=len(hallucinated) == 0,
            hallucinated_methods=hallucinated,
            used_methods=list(used),
            available_methods=list(available),
            confidence=confidence,
        )

    def _parse_context(self, context: str) -> set[str]:
        """Parse method names from context string.

        Handles context formats like:
        - "// Methods on DuckDBClient:"
        - "//   all<T>(sql): Promise<T[]>"
        - "// Available methods: all, run, prepare"

        Args:
            context: Raw context string from LSP

        Returns:
            Set of available method names
        """
        if not context:
            return set()

        methods = set()

        # Pattern 1: "// method_name(" or "//   method_name<"
        # Matches: "//   all<T>(sql)" -> "all"
        signature_pattern = r"//\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[<(]"
        for match in re.finditer(signature_pattern, context):
            methods.add(match.group(1))

        # Pattern 2: "// Available methods: method1, method2"
        list_pattern = r"//\s*(?:Available|VALID)\s+methods[^:]*:\s*([^\n]+)"
        for match in re.finditer(list_pattern, context, re.IGNORECASE):
            method_list = match.group(1)
            # Split by comma and clean
            for method in method_list.split(","):
                name = method.strip().split("(")[0].split("<")[0]
                if name and re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
                    methods.add(name)

        # Pattern 3: Just method names in comments "// methodName: ReturnType"
        simple_pattern = r"//\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:"
        for match in re.finditer(simple_pattern, context):
            methods.add(match.group(1))

        return methods

    def _extract_method_calls(self, code: str) -> set[str]:
        """Extract method call names from generated code.

        Looks for patterns like:
        - obj.methodName(
        - .methodName(
        - methodName(  (function calls)

        Args:
            code: Generated code to analyze

        Returns:
            Set of method/function names called in the code
        """
        if not code:
            return set()

        methods = set()

        # Pattern 1: Method calls after dot: "obj.method("
        dot_method_pattern = r"\.([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\(|<)"
        for match in re.finditer(dot_method_pattern, code):
            methods.add(match.group(1))

        # Pattern 2: Function-like calls that could be methods
        # e.g., "all(query)" - but only if it looks like a domain method
        if self.strict_mode:
            func_pattern = r"(?<![.\w])([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
            for match in re.finditer(func_pattern, code):
                name = match.group(1)
                # Skip obvious non-methods (keywords, constructors)
                if name not in {"if", "for", "while", "switch", "function", "class"}:
                    if not name[0].isupper():  # Skip PascalCase (constructors)
                        methods.add(name)

        return methods

    def suggest_corrections(
        self,
        hallucinated: list[str],
        available: list[str],
        max_suggestions: int = 3,
    ) -> dict[str, list[str]]:
        """Suggest corrections for hallucinated method names.

        Uses simple string similarity to suggest replacements.

        Args:
            hallucinated: List of hallucinated method names
            available: List of available methods
            max_suggestions: Max suggestions per hallucinated name

        Returns:
            Dict mapping hallucinated name to list of suggestions
        """
        suggestions = {}

        for bad_name in hallucinated:
            candidates = []
            for good_name in available:
                similarity = self._string_similarity(bad_name, good_name)
                if similarity > 0.4:  # Threshold for suggestion
                    candidates.append((good_name, similarity))

            # Sort by similarity and take top suggestions
            candidates.sort(key=lambda x: x[1], reverse=True)
            suggestions[bad_name] = [c[0] for c in candidates[:max_suggestions]]

        return suggestions

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate simple string similarity (Jaccard on character bigrams)."""
        if not s1 or not s2:
            return 0.0

        # Get character bigrams
        def bigrams(s: str) -> set[str]:
            return {s[i:i+2] for i in range(len(s) - 1)}

        b1 = bigrams(s1.lower())
        b2 = bigrams(s2.lower())

        if not b1 or not b2:
            return 1.0 if s1.lower() == s2.lower() else 0.0

        intersection = len(b1 & b2)
        union = len(b1 | b2)

        return intersection / union if union > 0 else 0.0


def validate_output(
    output: str,
    context: str,
    strict: bool = False,
) -> ValidationResult:
    """Convenience function for output validation.

    Args:
        output: Generated code to validate
        context: LSP context with available methods
        strict: Use strict validation mode

    Returns:
        ValidationResult
    """
    validator = OutputValidator(strict_mode=strict)
    return validator.validate(output, context)
