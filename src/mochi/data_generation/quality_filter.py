"""Automated quality filtering for training data.

Filters out low-quality training examples based on measurable criteria:
- Output length
- Syntactic completeness (bracket balance)
- Incomplete patterns (TODO, ...)
- Repetition detection
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Quality score for a training example."""

    total: float  # 0.0 - 1.0
    length_score: float
    balance_score: float
    completeness_score: float
    repetition_score: float
    reasons: list[str]  # Reasons for low score

    @property
    def is_acceptable(self) -> bool:
        """Check if the score meets minimum threshold."""
        return self.total >= 0.5


class QualityFilter:
    """Filter training data by quality metrics.

    Applies multiple heuristics to identify and filter out
    low-quality training examples.
    """

    def __init__(
        self,
        min_output_length: int = 20,
        max_output_length: int = 5000,
        min_score: float = 0.5,
    ) -> None:
        """Initialize quality filter.

        Args:
            min_output_length: Minimum output length in characters
            max_output_length: Maximum output length in characters
            min_score: Minimum quality score to accept (0.0 - 1.0)
        """
        self.min_output_length = min_output_length
        self.max_output_length = max_output_length
        self.min_score = min_score

    def score(self, example: dict[str, Any]) -> QualityScore:
        """Calculate quality score for a training example.

        Args:
            example: Training example with 'output' field (or 'text' for mlx format)

        Returns:
            QualityScore with detailed metrics
        """
        output = self._extract_output(example)
        reasons: list[str] = []

        # 1. Length check
        length_score = self._score_length(output, reasons)

        # 2. Bracket balance check
        balance_score = self._score_bracket_balance(output, reasons)

        # 3. Completeness check
        completeness_score = self._score_completeness(output, reasons)

        # 4. Repetition check
        repetition_score = self._score_repetition(output, reasons)

        # Calculate weighted total
        total = (
            length_score * 0.3
            + balance_score * 0.3
            + completeness_score * 0.2
            + repetition_score * 0.2
        )

        return QualityScore(
            total=total,
            length_score=length_score,
            balance_score=balance_score,
            completeness_score=completeness_score,
            repetition_score=repetition_score,
            reasons=reasons,
        )

    def _extract_output(self, example: dict[str, Any]) -> str:
        """Extract output from example (supports both formats)."""
        # mlx-lm format: {"text": "### Instruction:...\n### Response:\n<output>"}
        if "text" in example:
            text = example["text"]
            if "### Response:" in text:
                return text.split("### Response:")[-1].strip()
            return text

        # Alpaca format: {"output": "<output>"}
        return example.get("output", "")

    def _score_length(self, output: str, reasons: list[str]) -> float:
        """Score based on output length."""
        length = len(output)

        if length < self.min_output_length:
            reasons.append(f"Too short: {length} < {self.min_output_length}")
            return 0.3

        if length > self.max_output_length:
            reasons.append(f"Too long: {length} > {self.max_output_length}")
            return 0.7

        # Ideal range: 50-500 characters
        if 50 <= length <= 500:
            return 1.0
        elif length < 50:
            return 0.5 + (length / 100)  # 0.5-1.0
        else:
            return max(0.6, 1.0 - (length - 500) / 5000)  # Gradual decrease

    def _score_bracket_balance(self, output: str, reasons: list[str]) -> float:
        """Score based on bracket balance."""
        brackets = [
            ("{", "}"),
            ("(", ")"),
            ("[", "]"),
        ]

        total_balance = 0
        total_count = 0

        for open_b, close_b in brackets:
            open_count = output.count(open_b)
            close_count = output.count(close_b)

            if open_count > 0 or close_count > 0:
                total_count += 1
                if open_count == close_count:
                    total_balance += 1
                else:
                    diff = abs(open_count - close_count)
                    # Allow small imbalance for partial code
                    if diff <= 1:
                        total_balance += 0.7
                    elif diff <= 2:
                        total_balance += 0.4

        if total_count == 0:
            return 1.0  # No brackets to check

        score = total_balance / total_count

        if score < 0.7:
            reasons.append("Unbalanced brackets")

        return score

    def _score_completeness(self, output: str, reasons: list[str]) -> float:
        """Score based on completeness indicators."""
        score = 1.0

        # Check for incomplete patterns
        incomplete_patterns = [
            (r"\.\.\.$", "Ends with ..."),
            (r"//\s*TODO", "Contains TODO"),
            (r"//\s*FIXME", "Contains FIXME"),
            (r"//\s*XXX", "Contains XXX"),
            (r"\.\.\.\s*\}", "Incomplete block (...)"),
            (r"^\s*\.\.\.\s*$", "Only ellipsis", re.MULTILINE),
        ]

        for pattern, reason, *flags in incomplete_patterns:
            flag = flags[0] if flags else 0
            if re.search(pattern, output, flag):
                score -= 0.3
                reasons.append(reason)

        # Check for truncated output
        if output.rstrip().endswith((",", ":", "=", "+")):
            score -= 0.2
            reasons.append("Ends with incomplete token")

        return max(0.0, score)

    def _score_repetition(self, output: str, reasons: list[str]) -> float:
        """Score based on repetition detection."""
        lines = output.split("\n")

        if len(lines) < 3:
            return 1.0

        # Check for repeated lines
        seen_lines: dict[str, int] = {}
        for line in lines:
            stripped = line.strip()
            if len(stripped) > 10:  # Ignore short lines
                seen_lines[stripped] = seen_lines.get(stripped, 0) + 1

        # Calculate repetition ratio
        total_significant = sum(1 for l in lines if len(l.strip()) > 10)
        if total_significant == 0:
            return 1.0

        repeated = sum(1 for count in seen_lines.values() if count > 1)
        repetition_ratio = repeated / total_significant

        if repetition_ratio > 0.3:
            reasons.append(f"High repetition: {repetition_ratio:.1%}")
            return max(0.3, 1.0 - repetition_ratio)

        return 1.0

    def filter(
        self, examples: list[dict[str, Any]], verbose: bool = False
    ) -> list[dict[str, Any]]:
        """Filter examples by quality score.

        Args:
            examples: List of training examples
            verbose: Log filtering statistics

        Returns:
            Filtered list of high-quality examples
        """
        filtered: list[dict[str, Any]] = []
        rejected_reasons: dict[str, int] = {}

        for example in examples:
            score = self.score(example)

            if score.total >= self.min_score:
                filtered.append(example)
            else:
                # Track rejection reasons
                for reason in score.reasons:
                    rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1

        if verbose:
            logger.info(f"Quality filter: {len(filtered)}/{len(examples)} accepted")
            if rejected_reasons:
                logger.info("Rejection reasons:")
                for reason, count in sorted(
                    rejected_reasons.items(), key=lambda x: -x[1]
                ):
                    logger.info(f"  {reason}: {count}")

        return filtered

    def filter_with_stats(
        self, examples: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Filter examples and return statistics.

        Args:
            examples: List of training examples

        Returns:
            Tuple of (filtered_examples, statistics)
        """
        filtered: list[dict[str, Any]] = []
        scores: list[float] = []
        rejected_reasons: dict[str, int] = {}

        for example in examples:
            score = self.score(example)
            scores.append(score.total)

            if score.total >= self.min_score:
                filtered.append(example)
            else:
                for reason in score.reasons:
                    rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1

        stats = {
            "total": len(examples),
            "accepted": len(filtered),
            "rejected": len(examples) - len(filtered),
            "acceptance_rate": len(filtered) / len(examples) if examples else 0,
            "score_mean": sum(scores) / len(scores) if scores else 0,
            "score_min": min(scores) if scores else 0,
            "score_max": max(scores) if scores else 0,
            "rejection_reasons": rejected_reasons,
        }

        return filtered, stats


def filter_training_data(
    data: list[dict[str, Any]],
    min_output_length: int = 20,
    min_score: float = 0.5,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Convenience function to filter training data.

    Args:
        data: List of training examples
        min_output_length: Minimum output length
        min_score: Minimum quality score
        verbose: Log statistics

    Returns:
        Filtered list of examples
    """
    filter = QualityFilter(
        min_output_length=min_output_length,
        min_score=min_score,
    )
    return filter.filter(data, verbose=verbose)
