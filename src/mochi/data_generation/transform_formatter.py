"""Format code transformations as Alpaca training data.

Converts CodeTransformPair instances to Alpaca format compatible with
mlx-lm training.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .diff_extractor import CodeTransformPair
from .pattern_classifier import ClassificationResult


@dataclass
class TransformExample:
    """Alpaca format training example for code transformation."""

    instruction: str
    input: str  # Before code with context
    output: str  # After code (expected transformation)
    metadata: dict[str, Any]  # For tracking and debugging


class TransformAlpacaFormatter:
    """Format code transformations as Alpaca training data.

    Creates training examples in Alpaca format:
    - instruction: What transformation to perform
    - input: The code before transformation (with file path context)
    - output: The expected code after transformation

    Compatible with mlx-lm's "text" format for LoRA training.
    """

    # Instruction templates for each transform type (variety for robustness)
    INSTRUCTION_TEMPLATES: dict[str, list[str]] = {
        "error-handling": [
            "Add error handling to this function",
            "Wrap this code with try-catch and handle errors appropriately",
            "Add proper error handling following TypeScript best practices",
            "Handle potential errors in this code",
            "Add error recovery to this function",
            "Make this function handle exceptions properly",
            "Add try-catch blocks where needed",
        ],
        "null-safety": [
            "Add null safety checks to this code",
            "Make this code null-safe using optional chaining",
            "Add proper null checks to prevent runtime errors",
            "Handle null and undefined values safely",
            "Add defensive null handling",
            "Make this code handle missing values safely",
            "Add optional chaining and nullish coalescing",
        ],
        "type-safety": [
            "Add type annotations to this code",
            "Improve type safety of this function",
            "Add proper TypeScript types",
            "Add type guards where appropriate",
            "Strengthen the types in this code",
            "Add explicit type annotations",
            "Make this code more type-safe",
        ],
        "async-await": [
            "Convert this callback-based code to async/await",
            "Refactor to use async/await pattern",
            "Modernize this async code using await",
            "Convert Promise chains to async/await",
            "Rewrite using async/await syntax",
        ],
        "validation": [
            "Add input validation to this function",
            "Add runtime validation using zod",
            "Validate the input parameters",
            "Add schema validation to this code",
            "Add assertions to verify input",
            "Add parameter validation",
        ],
    }

    def __init__(self, vary_instructions: bool = True):
        """Initialize formatter.

        Args:
            vary_instructions: If True, randomly select from templates for variety
        """
        self.vary_instructions = vary_instructions

    def format(
        self,
        pair: CodeTransformPair,
        classification: ClassificationResult | None = None,
    ) -> TransformExample:
        """Format a single transformation pair as training example.

        Args:
            pair: The code transformation pair
            classification: Optional classification result with custom instruction

        Returns:
            TransformExample in Alpaca format
        """
        # Get instruction
        if classification and classification.instruction:
            instruction = classification.instruction
        else:
            instruction = self._get_instruction(pair.transform_type)

        # Format input with file path context
        input_text = f"// File: {pair.file_path}\n{pair.before_code}"

        # Output is the transformed code
        output_text = pair.after_code

        # Metadata for debugging and analysis
        metadata = {
            "transform_type": pair.transform_type,
            "commit": pair.commit_hash,
            "file_path": pair.file_path,
            "source": "git-diff",
        }

        if classification:
            metadata["confidence"] = classification.confidence
            metadata["classification_reason"] = classification.reason

        return TransformExample(
            instruction=instruction,
            input=input_text,
            output=output_text,
            metadata=metadata,
        )

    def format_batch(
        self,
        pairs: list[tuple[CodeTransformPair, ClassificationResult | None]],
    ) -> list[TransformExample]:
        """Format multiple transformation pairs.

        Args:
            pairs: List of (pair, classification) tuples

        Returns:
            List of TransformExample instances
        """
        examples = []
        for pair, classification in pairs:
            try:
                example = self.format(pair, classification)
                examples.append(example)
            except Exception as e:
                # Skip malformed pairs
                continue
        return examples

    def _get_instruction(self, transform_type: str) -> str:
        """Get instruction text for a transform type."""
        templates = self.INSTRUCTION_TEMPLATES.get(transform_type, ["Transform this code"])

        if self.vary_instructions:
            return random.choice(templates)
        return templates[0]

    def to_jsonl(
        self,
        examples: list[TransformExample],
        output_path: Path | str,
        format: str = "text",
    ) -> None:
        """Save examples to JSONL file.

        Args:
            examples: List of TransformExample instances
            output_path: Output file path
            format: Output format
                - "text": mlx-lm compatible single "text" field
                - "alpaca": Standard Alpaca format (instruction/input/output)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for example in examples:
                if format == "text":
                    # mlx-lm compatible format
                    text = (
                        f"### Instruction:\n{example.instruction}\n\n"
                        f"### Input:\n{example.input}\n\n"
                        f"### Response:\n{example.output}"
                    )
                    record = {"text": text}
                else:
                    # Standard Alpaca format
                    record = {
                        "instruction": example.instruction,
                        "input": example.input,
                        "output": example.output,
                    }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def to_json(
        self,
        examples: list[TransformExample],
        output_path: Path | str,
        include_metadata: bool = True,
    ) -> None:
        """Save examples to JSON file (for inspection/debugging).

        Args:
            examples: List of TransformExample instances
            output_path: Output file path
            include_metadata: Whether to include metadata in output
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        records = []
        for example in examples:
            record = {
                "instruction": example.instruction,
                "input": example.input,
                "output": example.output,
            }
            if include_metadata:
                record["metadata"] = example.metadata
            records.append(record)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)


def create_train_valid_split(
    examples: list[TransformExample],
    train_ratio: float = 0.9,
    shuffle: bool = True,
) -> tuple[list[TransformExample], list[TransformExample]]:
    """Split examples into train and validation sets.

    Args:
        examples: List of all examples
        train_ratio: Ratio of examples for training (default 0.9)
        shuffle: Whether to shuffle before splitting

    Returns:
        Tuple of (train_examples, valid_examples)
    """
    if shuffle:
        examples = examples.copy()
        random.shuffle(examples)

    split_idx = int(len(examples) * train_ratio)
    train_examples = examples[:split_idx]
    valid_examples = examples[split_idx:]

    return train_examples, valid_examples


def format_and_save(
    pairs: list[tuple[CodeTransformPair, ClassificationResult | None]],
    output_dir: Path | str,
    train_ratio: float = 0.9,
    format: str = "text",
) -> tuple[Path, Path]:
    """Format pairs and save to train/valid files.

    Args:
        pairs: List of (pair, classification) tuples
        output_dir: Output directory
        train_ratio: Ratio for train/valid split
        format: Output format ("text" or "alpaca")

    Returns:
        Tuple of (train_path, valid_path)
    """
    formatter = TransformAlpacaFormatter()
    examples = formatter.format_batch(pairs)

    train_examples, valid_examples = create_train_valid_split(
        examples, train_ratio=train_ratio
    )

    output_dir = Path(output_dir)
    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"

    formatter.to_jsonl(train_examples, train_path, format=format)
    formatter.to_jsonl(valid_examples, valid_path, format=format)

    return train_path, valid_path
