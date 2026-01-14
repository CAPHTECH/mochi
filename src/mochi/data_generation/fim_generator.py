"""Fill-in-the-Middle (FIM) data generator.

Generates FIM variants from existing training examples.
This enables models to learn the <FILL> marker format used by complete_code tool.

The FIM format:
    Instruction: "Fill in the code at <FILL> marker. Output only the code to insert, no explanations."
    Input: "{prefix}<FILL>{suffix}"
    Response: "{fill_content}"
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Generator


@dataclass
class FIMExample:
    """A Fill-in-the-Middle training example."""

    prefix: str
    suffix: str
    fill_content: str

    def to_training_format(self) -> dict[str, str]:
        """Convert to mlx-lm training format."""
        instruction = "Fill in the code at <FILL> marker. Output only the code to insert, no explanations."
        input_text = f"{self.prefix}<FILL>{self.suffix}"

        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{self.fill_content}"
        )
        return {"text": text}


class FIMGenerator:
    """Generator for Fill-in-the-Middle training examples.

    Takes existing training examples and creates FIM variants
    by finding logical cut points in the response code.

    Usage:
        generator = FIMGenerator(fim_ratio=0.3)
        fim_examples = generator.generate_from_examples(examples)
    """

    # Patterns that indicate good cut points (after these)
    CUT_AFTER_PATTERNS = [
        r'\{\s*$',           # After opening brace
        r';\s*$',            # After semicolon
        r',\s*$',            # After comma
        r'=>\s*\{\s*$',      # After arrow function opening
        r'\(\s*$',           # After opening paren
    ]

    # Patterns that indicate good end points (before these)
    CUT_BEFORE_PATTERNS = [
        r'^\s*\}',           # Before closing brace
        r'^\s*\)',           # Before closing paren
        r'^\s*\],',          # Before closing bracket with comma
    ]

    # Minimum lengths for meaningful FIM
    MIN_PREFIX_LENGTH = 30
    MIN_SUFFIX_LENGTH = 20
    MIN_FILL_LENGTH = 10
    MAX_FILL_LENGTH = 500

    def __init__(
        self,
        fim_ratio: float = 0.3,
        max_variants_per_example: int = 2,
        seed: int | None = None,
    ) -> None:
        """Initialize FIM generator.

        Args:
            fim_ratio: Ratio of FIM examples to generate (relative to input)
            max_variants_per_example: Maximum FIM variants per source example
            seed: Random seed for reproducibility
        """
        self.fim_ratio = fim_ratio
        self.max_variants_per_example = max_variants_per_example
        self.rng = random.Random(seed)

    def generate_from_examples(
        self,
        examples: list[dict],
    ) -> list[dict]:
        """Generate FIM examples from existing training examples.

        Args:
            examples: List of training examples in mlx-lm format

        Returns:
            List of FIM training examples
        """
        fim_examples = []
        target_count = int(len(examples) * self.fim_ratio)

        # Shuffle to get diverse samples
        shuffled = list(examples)
        self.rng.shuffle(shuffled)

        for example in shuffled:
            if len(fim_examples) >= target_count:
                break

            variants = list(self._generate_variants(example))
            fim_examples.extend(variants[:self.max_variants_per_example])

        return fim_examples

    def _generate_variants(
        self,
        example: dict,
    ) -> Generator[dict, None, None]:
        """Generate FIM variants for a single example.

        Args:
            example: Training example in mlx-lm format

        Yields:
            FIM training examples
        """
        text = example.get("text", "")

        # Extract input and response
        if "### Response:" not in text:
            return

        parts = text.split("### Response:", 1)
        if len(parts) != 2:
            return

        header = parts[0]  # Contains instruction and input
        response = parts[1].strip()

        # Extract the input section
        if "### Input:" in header:
            input_parts = header.split("### Input:", 1)
            input_code = input_parts[1].strip()
        else:
            input_code = ""

        if not response or len(response) < self.MIN_FILL_LENGTH + self.MIN_SUFFIX_LENGTH:
            return

        # Find cut points
        cut_points = self._find_cut_points(response)

        if not cut_points:
            return

        # Generate variants from cut points
        for start_idx, end_idx in cut_points:
            fill_content = response[start_idx:end_idx].strip()

            # Validate lengths
            if len(fill_content) < self.MIN_FILL_LENGTH:
                continue
            if len(fill_content) > self.MAX_FILL_LENGTH:
                continue

            prefix = input_code + response[:start_idx]
            suffix = response[end_idx:]

            if len(prefix) < self.MIN_PREFIX_LENGTH:
                continue
            if len(suffix) < self.MIN_SUFFIX_LENGTH:
                continue

            fim_example = FIMExample(
                prefix=prefix,
                suffix=suffix,
                fill_content=fill_content,
            )
            yield fim_example.to_training_format()

    def _find_cut_points(self, code: str) -> list[tuple[int, int]]:
        """Find logical cut points in code.

        Returns list of (start_idx, end_idx) tuples for potential fills.
        """
        cut_points = []
        lines = code.split('\n')

        # Track positions
        current_pos = 0
        potential_starts = []

        for i, line in enumerate(lines):
            line_start = current_pos
            line_end = current_pos + len(line)

            # Check for cut-after patterns
            for pattern in self.CUT_AFTER_PATTERNS:
                if re.search(pattern, line):
                    potential_starts.append(line_end + 1)  # After newline

            # Check for cut-before patterns
            for pattern in self.CUT_BEFORE_PATTERNS:
                if re.search(pattern, line):
                    # Try to pair with a previous start
                    for start in potential_starts:
                        if line_start - start >= self.MIN_FILL_LENGTH:
                            cut_points.append((start, line_start))

            current_pos = line_end + 1  # +1 for newline

        # Also try statement-based cuts (between semicolons)
        statement_cuts = self._find_statement_cuts(code)
        cut_points.extend(statement_cuts)

        # Deduplicate and sort
        cut_points = list(set(cut_points))
        cut_points.sort(key=lambda x: x[0])

        return cut_points

    def _find_statement_cuts(self, code: str) -> list[tuple[int, int]]:
        """Find cut points based on statement boundaries."""
        cuts = []

        # Find all semicolon positions
        semicolons = [m.end() for m in re.finditer(r';\s*\n', code)]

        if len(semicolons) < 2:
            return cuts

        # Try consecutive pairs
        for i in range(len(semicolons) - 1):
            start = semicolons[i]
            for j in range(i + 1, min(i + 4, len(semicolons))):  # Limit range
                end = semicolons[j]
                if end - start >= self.MIN_FILL_LENGTH:
                    cuts.append((start, end))

        return cuts


def generate_fim_examples(
    examples: list[dict],
    fim_ratio: float = 0.3,
    max_variants: int = 2,
    seed: int | None = None,
) -> list[dict]:
    """Generate FIM training examples from existing examples.

    Convenience function for generating FIM variants.

    Args:
        examples: Source training examples
        fim_ratio: Ratio of FIM examples to generate
        max_variants: Maximum variants per source example
        seed: Random seed

    Returns:
        List of FIM training examples
    """
    generator = FIMGenerator(
        fim_ratio=fim_ratio,
        max_variants_per_example=max_variants,
        seed=seed,
    )
    return generator.generate_from_examples(examples)
