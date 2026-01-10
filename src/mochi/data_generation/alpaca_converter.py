"""Convert code chunks to Alpaca training format."""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from mochi.preprocessing.code_chunker import CodeChunk


@dataclass
class AlpacaExample:
    """Alpaca format training example."""

    instruction: str
    input: str  # context
    output: str  # expected completion


class AlpacaConverter:
    """Convert code chunks to Alpaca format for training."""

    COMPLETION_TEMPLATES = [
        "Complete the following {language} code:\n{context}",
        "Write the implementation for this {language} {chunk_type}:\n{context}",
        "Implement the following {language} code based on the context:\n{context}",
        "Fill in the {language} code:\n{context}",
    ]

    EXPLANATION_TEMPLATES = [
        "Explain what this {language} code does:\n{code}",
        "Describe the purpose of this {language} {chunk_type}:\n{code}",
        "What does this {language} function/class do?\n{code}",
    ]

    def __init__(self, project_name: str = "project") -> None:
        """Initialize converter with project name for context."""
        self.project_name = project_name

    def convert_chunks(
        self,
        chunks: list[CodeChunk],
        include_completion: bool = True,
        include_explanation: bool = True,
        completion_ratio: float = 0.5,
    ) -> list[AlpacaExample]:
        """
        Convert code chunks to Alpaca training examples.

        Args:
            chunks: List of code chunks
            include_completion: Generate code completion examples
            include_explanation: Generate code explanation examples
            completion_ratio: For completion tasks, ratio of code to show as context
        """
        examples: list[AlpacaExample] = []

        for chunk in chunks:
            if include_completion:
                examples.extend(
                    self._create_completion_examples(chunk, completion_ratio)
                )
            if include_explanation:
                examples.extend(self._create_explanation_examples(chunk))

        return examples

    def _create_completion_examples(
        self,
        chunk: CodeChunk,
        completion_ratio: float,
    ) -> list[AlpacaExample]:
        """Create code completion training examples."""
        examples: list[AlpacaExample] = []
        lines = chunk.content.split("\n")

        if len(lines) < 3:
            return examples

        # Create completion example at different points
        split_points = [
            int(len(lines) * 0.3),
            int(len(lines) * 0.5),
            int(len(lines) * 0.7),
        ]

        for split_point in split_points:
            if split_point < 1 or split_point >= len(lines) - 1:
                continue

            context_lines = lines[:split_point]
            completion_lines = lines[split_point:]

            context = "\n".join(context_lines)
            completion = "\n".join(completion_lines)

            # Add file context if available
            full_context = ""
            if chunk.context:
                full_context = f"// File: {chunk.source_file}\n{chunk.context}\n\n"
            full_context += context

            template = random.choice(self.COMPLETION_TEMPLATES)
            instruction = template.format(
                language=chunk.language,
                chunk_type=chunk.chunk_type,
                context="",
            )

            examples.append(
                AlpacaExample(
                    instruction=instruction,
                    input=full_context,
                    output=completion,
                )
            )

        return examples

    def _create_explanation_examples(self, chunk: CodeChunk) -> list[AlpacaExample]:
        """Create code explanation training examples."""
        # For explanation, we'd ideally have docstrings or comments
        # For MVP, we create a simple template
        template = random.choice(self.EXPLANATION_TEMPLATES)
        instruction = template.format(
            language=chunk.language,
            chunk_type=chunk.chunk_type,
            code="",
        )

        # Extract docstring/comment if present for the output
        explanation = self._extract_documentation(chunk)
        if not explanation:
            # Skip if no documentation available
            return []

        return [
            AlpacaExample(
                instruction=instruction,
                input=chunk.content,
                output=explanation,
            )
        ]

    def _extract_documentation(self, chunk: CodeChunk) -> str | None:
        """Extract documentation from code chunk."""
        lines = chunk.content.split("\n")

        # Look for JSDoc style comments
        doc_lines: list[str] = []
        in_doc = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("/**"):
                in_doc = True
                doc_lines.append(stripped)
            elif in_doc:
                doc_lines.append(stripped)
                if stripped.endswith("*/"):
                    break

        if doc_lines:
            # Clean up the doc
            doc = "\n".join(doc_lines)
            doc = doc.replace("/**", "").replace("*/", "").replace("*", "").strip()
            if len(doc) > 10:  # Meaningful documentation
                return doc

        return None

    def to_jsonl(self, examples: list[AlpacaExample], output_path: str | Path) -> None:
        """Save examples to JSONL file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")

    def to_json(self, examples: list[AlpacaExample], output_path: str | Path) -> None:
        """Save examples to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in examples], f, ensure_ascii=False, indent=2)


def create_training_dataset(
    chunks: list[CodeChunk],
    output_dir: str | Path,
    project_name: str = "project",
    train_ratio: float = 0.9,
) -> tuple[Path, Path]:
    """
    Create train/eval split and save to files.

    Returns:
        Tuple of (train_path, eval_path)
    """
    converter = AlpacaConverter(project_name)
    examples = converter.convert_chunks(chunks)

    # Shuffle and split
    random.shuffle(examples)
    split_idx = int(len(examples) * train_ratio)

    train_examples = examples[:split_idx]
    eval_examples = examples[split_idx:]

    output_dir = Path(output_dir)
    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"

    converter.to_jsonl(train_examples, train_path)
    converter.to_jsonl(eval_examples, eval_path)

    return train_path, eval_path
