"""Convert code chunks to Alpaca training format.

Supports optional LSP-based context extraction for improved training data
with accurate type and method information.

Law compliance:
- L-fallback-graceful: Works without LSP context (graceful degradation)
- L-context-format: Uses "// Available methods: ..." format when context available
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mochi.preprocessing.code_chunker import CodeChunk

if TYPE_CHECKING:
    from mochi.lsp.context_extractor import ContextExtractor

logger = logging.getLogger(__name__)


@dataclass
class AlpacaExample:
    """Alpaca format training example."""

    instruction: str
    input: str  # context
    output: str  # expected completion


class AlpacaConverter:
    """Convert code chunks to Alpaca format for training.

    Optionally integrates with LSP-based ContextExtractor to add accurate
    type and method information to training examples.

    Terms:
    - ContextBlock: LSP-derived context (methods, types, schema)
    - AlpacaExample: Training example with instruction/input/output

    Laws:
    - L-fallback-graceful: Works without context_extractor
    """

    # Task type templates for diverse training data
    COMPLETION_TEMPLATES = [
        "Complete the following {language} code:",
        "Write the implementation for this {language} {chunk_type}:",
        "Implement the following {language} code based on the context:",
        "Fill in the {language} code:",
        "Continue this {language} code:",
    ]

    EXPLANATION_TEMPLATES = [
        "Explain what this {language} code does:",
        "Describe the purpose of this {language} {chunk_type}:",
        "What does this {language} function/class do?",
        "Summarize the functionality of this code:",
    ]

    # New instruction types for diverse training
    REFACTOR_TEMPLATES = [
        "Refactor this {language} code to improve readability:",
        "Simplify this {language} {chunk_type}:",
        "Improve the structure of this {language} code:",
        "Clean up this {language} code:",
    ]

    TYPE_ANNOTATION_TEMPLATES = [
        "Add type annotations to this {language} code:",
        "Annotate the types in this {language} {chunk_type}:",
        "Add proper type hints to the following code:",
    ]

    ERROR_HANDLING_TEMPLATES = [
        "Add error handling to this {language} code:",
        "Improve the error handling in this {chunk_type}:",
        "Add try-catch blocks where appropriate:",
    ]

    DOCSTRING_TEMPLATES = [
        "Add documentation comments to this {language} code:",
        "Write JSDoc/docstring for this {chunk_type}:",
        "Document this {language} function/class:",
    ]

    # Method calling templates (for learning API patterns)
    METHOD_CALL_TEMPLATES = [
        "Complete the method call in this {language} code:",
        "What method should be called here?",
        "Fill in the appropriate API call:",
    ]

    # Import statement templates
    IMPORT_TEMPLATES = [
        "Add the necessary import statements for this {language} code:",
        "What imports are needed for this code?",
        "Complete the import statements:",
    ]

    # P1: Type-aware completion templates (teach model to use typed context)
    TYPE_AWARE_TEMPLATES = [
        "Given {var_name} is of type {type_name}, complete the method call:",
        "Complete the code. {var_name} is a {type_name}:",
        "Using the {type_name} API, complete:",
        "The variable {var_name} ({type_name}) needs a method call:",
    ]

    def __init__(
        self,
        project_name: str = "project",
        context_extractor: ContextExtractor | None = None,
        project_root: Path | None = None,
    ) -> None:
        """Initialize converter with optional LSP context extractor.

        Args:
            project_name: Project name for context
            context_extractor: Optional LSP-based context extractor for
                             adding type/method information to examples
            project_root: Root directory for resolving relative file paths
        """
        self.project_name = project_name
        self.context_extractor = context_extractor
        self.project_root = project_root

    def convert_chunks(
        self,
        chunks: list[CodeChunk],
        include_completion: bool = True,
        include_explanation: bool = True,
        include_method_call: bool = True,
        include_docstring: bool = True,
        include_type_aware: bool = True,  # P1: 型認識例
        completion_ratio: float = 0.5,
    ) -> list[AlpacaExample]:
        """
        Convert code chunks to Alpaca training examples.

        Args:
            chunks: List of code chunks
            include_completion: Generate code completion examples
            include_explanation: Generate code explanation examples
            include_method_call: Generate method call completion examples
            include_docstring: Generate documentation examples
            include_type_aware: Generate type-aware examples (P1)
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
            if include_method_call:
                examples.extend(self._create_method_call_examples(chunk))
            if include_docstring:
                examples.extend(self._create_docstring_examples(chunk))
            if include_type_aware:
                examples.extend(self._create_type_aware_examples(chunk))

        return examples

    async def convert_chunks_async(
        self,
        chunks: list[CodeChunk],
        include_completion: bool = True,
        include_explanation: bool = True,
        include_method_call: bool = True,
        include_docstring: bool = True,
        include_type_aware: bool = True,  # P1: 型認識例
        completion_ratio: float = 0.5,
    ) -> list[AlpacaExample]:
        """Convert code chunks to Alpaca examples with LSP context.

        Async version that extracts LSP context for each chunk when
        context_extractor is available.

        Args:
            chunks: List of code chunks
            include_completion: Generate code completion examples
            include_explanation: Generate code explanation examples
            include_method_call: Generate method call completion examples
            include_docstring: Generate documentation examples
            include_type_aware: Generate type-aware examples (P1)
            completion_ratio: Ratio of code to show as context

        Returns:
            List of AlpacaExample with LSP context included
        """
        examples: list[AlpacaExample] = []

        for chunk in chunks:
            # Extract LSP context if available
            lsp_context = ""
            if self.context_extractor and include_completion:
                try:
                    lsp_context = await self._get_lsp_context(chunk)
                except Exception as e:
                    # L-fallback-graceful: Log and continue without context
                    logger.warning(
                        f"LSP context extraction failed for {chunk.source_file}: {e}"
                    )

            if include_completion:
                examples.extend(
                    self._create_completion_examples(
                        chunk, completion_ratio, lsp_context
                    )
                )
            if include_explanation:
                examples.extend(self._create_explanation_examples(chunk))
            if include_method_call:
                examples.extend(
                    self._create_method_call_examples(chunk, lsp_context)
                )
            if include_docstring:
                examples.extend(self._create_docstring_examples(chunk))
            if include_type_aware:
                examples.extend(
                    self._create_type_aware_examples(chunk, lsp_context)
                )

        return examples

    async def _get_lsp_context(self, chunk: CodeChunk) -> str:
        """Get LSP context for a code chunk.

        Extracts context at the midpoint of the chunk for representative
        completions.
        """
        if not self.context_extractor:
            return ""

        # Resolve file path (may be relative to project_root)
        file_path = Path(chunk.source_file)
        if not file_path.is_absolute() and self.project_root:
            file_path = self.project_root / file_path

        if not file_path.exists():
            return ""

        # Get context at the midpoint of the chunk
        lines = chunk.content.split("\n")
        mid_line = chunk.start_line + len(lines) // 2

        # Find a good position (after a dot or at line start)
        context_block = await self.context_extractor.extract_at_position(
            file_path, mid_line, 0
        )

        return context_block.format() if not context_block.is_empty() else ""

    def _create_completion_examples(
        self,
        chunk: CodeChunk,
        completion_ratio: float,
        lsp_context: str = "",
    ) -> list[AlpacaExample]:
        """Create code completion training examples.

        Args:
            chunk: Source code chunk
            completion_ratio: Ratio of code to show as context
            lsp_context: Optional LSP-derived context block
        """
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

            # Build full context with LSP info if available
            full_context_parts = []

            # Add file path
            if chunk.source_file:
                full_context_parts.append(f"// File: {chunk.source_file}")

            # Add LSP context block (methods, types, schema)
            if lsp_context:
                full_context_parts.append(lsp_context)

            # Add chunk context if available
            if chunk.context:
                full_context_parts.append(chunk.context)

            # Add separator and code context
            if full_context_parts:
                full_context = "\n".join(full_context_parts) + "\n\n" + context
            else:
                full_context = context

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

    def _create_method_call_examples(
        self,
        chunk: CodeChunk,
        lsp_context: str = "",
    ) -> list[AlpacaExample]:
        """Create method call completion examples.

        These examples focus on completing method calls, which is crucial
        for learning API patterns and correct method usage.
        """
        examples: list[AlpacaExample] = []
        lines = chunk.content.split("\n")

        # Find lines with method calls (containing dots)
        for i, line in enumerate(lines):
            if "." not in line:
                continue

            # Find the position of the dot
            dot_positions = [j for j, c in enumerate(line) if c == "."]
            if not dot_positions:
                continue

            # Take the first meaningful dot (not at the start)
            for dot_pos in dot_positions:
                if dot_pos < 2:
                    continue

                # Create context: lines before + partial line up to and including dot
                context_lines = lines[:i]
                partial_line = line[: dot_pos + 1]
                context_lines.append(partial_line)

                # Expected completion: rest of the line
                completion = line[dot_pos + 1 :].strip()
                if not completion or len(completion) < 2:
                    continue

                # Build full context
                full_context_parts = []
                if chunk.source_file:
                    full_context_parts.append(f"// File: {chunk.source_file}")
                if lsp_context:
                    full_context_parts.append(lsp_context)

                context = "\n".join(context_lines)
                if full_context_parts:
                    full_context = "\n".join(full_context_parts) + "\n\n" + context
                else:
                    full_context = context

                template = random.choice(self.METHOD_CALL_TEMPLATES)
                instruction = template.format(
                    language=chunk.language,
                    chunk_type=chunk.chunk_type,
                )

                examples.append(
                    AlpacaExample(
                        instruction=instruction,
                        input=full_context,
                        output=completion,
                    )
                )
                break  # One example per line

        # Limit to avoid too many similar examples
        # Increased to 10 for better API pattern learning (was 3)
        return examples[:10]

    def _create_type_aware_examples(
        self,
        chunk: CodeChunk,
        lsp_context: str = "",
    ) -> list[AlpacaExample]:
        """Create type-aware training examples.

        P1: 型認識例の追加 - モデルに型情報を活用させる学習

        These examples explicitly show the type of variables and require
        the model to use appropriate methods for that type.
        """
        examples: list[AlpacaExample] = []
        lines = chunk.content.split("\n")

        # Pattern to find typed variable declarations
        # e.g., "const db: DuckDBClient = ..." or "let users: User[] = ..."
        typed_var_pattern = re.compile(
            r'(?:const|let|var)\s+(\w+)\s*:\s*([A-Z][a-zA-Z0-9_<>\[\]]+)'
        )

        for i, line in enumerate(lines):
            match = typed_var_pattern.search(line)
            if not match:
                continue

            var_name = match.group(1)
            type_name = match.group(2)

            # Clean up type name (remove generics for simpler reference)
            simple_type = re.sub(r'<[^>]+>', '', type_name).replace('[]', '')

            # Find method calls on this variable in subsequent lines
            for j in range(i + 1, min(i + 10, len(lines))):
                if f"{var_name}." in lines[j]:
                    # Found a method call
                    dot_pos = lines[j].find(f"{var_name}.")
                    if dot_pos < 0:
                        continue

                    # Context: lines up to the dot
                    context_lines = lines[:j]
                    partial_line = lines[j][:dot_pos + len(var_name) + 1]
                    context_lines.append(partial_line)

                    # Expected: rest of line after the dot
                    completion = lines[j][dot_pos + len(var_name) + 1:].strip()
                    if not completion or len(completion) < 2:
                        continue

                    # Build context with type info
                    full_context_parts = []
                    if chunk.source_file:
                        full_context_parts.append(f"// File: {chunk.source_file}")

                    # Add explicit type context
                    full_context_parts.append(f"// {var_name} is of type: {type_name}")

                    if lsp_context:
                        full_context_parts.append(lsp_context)

                    context = "\n".join(context_lines)
                    if full_context_parts:
                        full_context = "\n".join(full_context_parts) + "\n\n" + context
                    else:
                        full_context = context

                    template = random.choice(self.TYPE_AWARE_TEMPLATES)
                    instruction = template.format(
                        var_name=var_name,
                        type_name=simple_type,
                        language=chunk.language,
                    )

                    examples.append(
                        AlpacaExample(
                            instruction=instruction,
                            input=full_context,
                            output=completion,
                        )
                    )
                    break  # One example per typed variable

        return examples[:5]  # Limit per chunk

    def _create_docstring_examples(self, chunk: CodeChunk) -> list[AlpacaExample]:
        """Create documentation generation examples.

        These examples teach the model to generate documentation
        from code, which helps with understanding code structure.
        """
        # Only for function/class chunks with documentation
        doc = self._extract_documentation(chunk)
        if not doc:
            return []

        # Remove documentation from the code for the input
        lines = chunk.content.split("\n")
        code_without_doc = []
        in_doc = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("/**"):
                in_doc = True
                continue
            if in_doc:
                if stripped.endswith("*/"):
                    in_doc = False
                continue
            code_without_doc.append(line)

        if len(code_without_doc) < 2:
            return []

        code_input = "\n".join(code_without_doc).strip()
        if not code_input:
            return []

        template = random.choice(self.DOCSTRING_TEMPLATES)
        instruction = template.format(
            language=chunk.language,
            chunk_type=chunk.chunk_type,
        )

        return [
            AlpacaExample(
                instruction=instruction,
                input=code_input,
                output=doc,
            )
        ]

    def _create_import_examples(
        self,
        chunk: CodeChunk,
        lsp_context: str = "",
    ) -> list[AlpacaExample]:
        """Create import statement completion examples.

        These examples teach the model to add necessary imports
        based on the code usage.
        """
        lines = chunk.content.split("\n")

        # Find import lines
        import_lines = []
        code_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                import_lines.append(line)
            elif stripped and not stripped.startswith("//") and not stripped.startswith("#"):
                code_lines.append(line)

        if not import_lines or len(code_lines) < 3:
            return []

        # Input: code without imports (first 10 lines)
        code_sample = "\n".join(code_lines[:10])

        # Output: the import statements
        imports_output = "\n".join(import_lines)

        # Build context
        full_context_parts = []
        if chunk.source_file:
            full_context_parts.append(f"// File: {chunk.source_file}")
        if lsp_context:
            full_context_parts.append(lsp_context)

        if full_context_parts:
            full_context = "\n".join(full_context_parts) + "\n\n" + code_sample
        else:
            full_context = code_sample

        template = random.choice(self.IMPORT_TEMPLATES)
        instruction = template.format(
            language=chunk.language,
            chunk_type=chunk.chunk_type,
        )

        return [
            AlpacaExample(
                instruction=instruction,
                input=full_context,
                output=imports_output,
            )
        ]

    def to_jsonl(
        self,
        examples: list[AlpacaExample],
        output_path: str | Path,
        format: str = "text",
    ) -> None:
        """Save examples to JSONL file.

        Args:
            examples: List of AlpacaExample instances
            output_path: Output file path
            format: Output format - "text" (mlx-lm compatible) or "alpaca"
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for example in examples:
                if format == "text":
                    # mlx-lm compatible format: single "text" field with full prompt
                    if example.input:
                        text = (
                            f"### Instruction:\n{example.instruction}\n\n"
                            f"### Input:\n{example.input}\n\n"
                            f"### Response:\n{example.output}"
                        )
                    else:
                        text = (
                            f"### Instruction:\n{example.instruction}\n\n"
                            f"### Response:\n{example.output}"
                        )
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                else:
                    # Original Alpaca format
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
    extra_examples: list[AlpacaExample] | None = None,
) -> tuple[Path, Path]:
    """
    Create train/valid split and save to files.

    Args:
        chunks: List of code chunks
        output_dir: Output directory for train/valid files
        project_name: Project name for context
        train_ratio: Train/valid split ratio
        extra_examples: Additional examples to include (e.g., package docs)

    Returns:
        Tuple of (train_path, valid_path)
    """
    converter = AlpacaConverter(project_name)
    examples = converter.convert_chunks(chunks)

    # Add extra examples (e.g., package documentation)
    if extra_examples:
        examples.extend(extra_examples)

    # Shuffle and split
    random.shuffle(examples)
    split_idx = int(len(examples) * train_ratio)

    train_examples = examples[:split_idx]
    valid_examples = examples[split_idx:]

    output_dir = Path(output_dir)
    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"

    converter.to_jsonl(train_examples, train_path)
    converter.to_jsonl(valid_examples, valid_path)

    return train_path, valid_path


def create_package_doc_examples(
    package_docs: list,  # list[PackageDoc]
    project_name: str = "project",
) -> list[AlpacaExample]:
    """Create training examples from package documentation.

    Converts package READMEs into training examples that help the model
    understand library APIs and usage patterns.

    Args:
        package_docs: List of PackageDoc from package_docs extractor
        project_name: Project name for context

    Returns:
        List of AlpacaExample for training
    """
    examples: list[AlpacaExample] = []

    # Templates for package documentation learning
    PACKAGE_TEMPLATES = [
        "How do I use {package_name} in {project_name}?",
        "What are the main features of {package_name}?",
        "Show me how to use {package_name}:",
        "Explain the {package_name} API:",
        "What patterns does {package_name} provide?",
    ]

    for doc in package_docs:
        if not doc.readme or len(doc.readme) < 100:
            continue

        # Create multiple examples per package
        for template in PACKAGE_TEMPLATES[:3]:  # Limit to 3 per package
            instruction = template.format(
                package_name=doc.name,
                project_name=project_name,
            )

            # Use the README as the output (what the model should learn)
            # Truncate if too long
            readme_content = doc.readme
            if len(readme_content) > 4000:
                readme_content = readme_content[:4000] + "\n\n[...]"

            examples.append(
                AlpacaExample(
                    instruction=instruction,
                    input=f"Package: {doc.name}\nVersion: {doc.version or 'latest'}",
                    output=readme_content,
                )
            )

    return examples


async def create_training_dataset_with_context(
    chunks: list[CodeChunk],
    output_dir: str | Path,
    project_root: str | Path,
    project_name: str = "project",
    language: str = "typescript",
    schema_path: Path | None = None,
    train_ratio: float = 0.9,
) -> tuple[Path, Path]:
    """Create train/valid split with LSP context and save to files.

    Enhanced version that uses LSP to extract accurate type and method
    information for training examples.

    Args:
        chunks: List of code chunks
        output_dir: Output directory for train/valid files
        project_root: Root directory of the project for LSP
        project_name: Project name for context
        language: Programming language (typescript, python, etc.)
        schema_path: Optional path to schema.yaml
        train_ratio: Train/valid split ratio

    Returns:
        Tuple of (train_path, valid_path)
    """
    from mochi.lsp.context_extractor import create_context_extractor

    # Create context extractor with LSP
    try:
        context_extractor = await create_context_extractor(
            project_root=Path(project_root),
            language=language,
            schema_path=schema_path,
        )
    except Exception as e:
        logger.warning(f"Failed to create LSP context extractor: {e}")
        context_extractor = None

    try:
        converter = AlpacaConverter(
            project_name,
            context_extractor=context_extractor,
            project_root=Path(project_root),
        )
        examples = await converter.convert_chunks_async(chunks)
    finally:
        # Cleanup LSP connection
        if context_extractor:
            await context_extractor.lsp.stop()

    # Shuffle and split
    random.shuffle(examples)
    split_idx = int(len(examples) * train_ratio)

    train_examples = examples[:split_idx]
    valid_examples = examples[split_idx:]

    output_dir = Path(output_dir)
    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"

    converter.to_jsonl(train_examples, train_path)
    converter.to_jsonl(valid_examples, valid_path)

    return train_path, valid_path
