"""AST-based extraction of complete code patterns.

Uses tree-sitter to extract syntactically complete code units:
- Functions (with full body)
- Classes (with all methods)
- React components
- State machine definitions
- Hook implementations

Ensures extracted patterns are syntactically valid and complete.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import tree_sitter_typescript as ts_typescript
from tree_sitter import Language, Node, Parser

logger = logging.getLogger(__name__)


@dataclass
class ExtractedPattern:
    """A complete code pattern extracted from source."""

    source_file: str
    pattern_type: str  # function, class, component, hook, machine, etc.
    name: str
    full_code: str
    signature: str  # Function signature or class declaration
    body: str  # Implementation body
    imports: list[str]  # Related imports
    start_line: int
    end_line: int
    language: str


class ASTExtractor:
    """Extract complete code patterns using AST analysis.

    Provides higher-quality training data by ensuring:
    - Syntactic completeness (balanced brackets)
    - Semantic completeness (full function bodies)
    - Context preservation (related imports)
    """

    # Pattern types and their node types
    PATTERN_TYPES = {
        "function": [
            "function_declaration",
            "arrow_function",
            "function_expression",
        ],
        "class": ["class_declaration"],
        "method": ["method_definition"],
        "component": ["function_declaration", "arrow_function"],  # React components
        "hook": ["function_declaration", "arrow_function"],  # Custom hooks
        "machine": ["variable_declaration"],  # XState machines
    }

    def __init__(self, language: str = "typescript") -> None:
        """Initialize extractor with language parser.

        Args:
            language: Programming language (typescript, tsx)
        """
        self.language = language
        self._setup_parser()

    def _setup_parser(self) -> None:
        """Set up tree-sitter parser."""
        if self.language in ("typescript", "javascript"):
            lang = Language(ts_typescript.language_typescript())
        elif self.language == "tsx":
            lang = Language(ts_typescript.language_tsx())
        else:
            raise ValueError(f"Unsupported language: {self.language}")

        self.parser = Parser(lang)

    def extract_from_file(
        self,
        file_path: Path,
        pattern_types: list[str] | None = None,
    ) -> list[ExtractedPattern]:
        """Extract patterns from a source file.

        Args:
            file_path: Path to source file
            pattern_types: Types of patterns to extract (None = all)

        Returns:
            List of extracted patterns
        """
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return []

        return self.extract_from_content(
            content=content,
            source_file=str(file_path),
            pattern_types=pattern_types,
        )

    def extract_from_content(
        self,
        content: str,
        source_file: str = "unknown",
        pattern_types: list[str] | None = None,
    ) -> list[ExtractedPattern]:
        """Extract patterns from source content.

        Args:
            content: Source code content
            source_file: Source file path for reference
            pattern_types: Types of patterns to extract

        Returns:
            List of extracted patterns
        """
        tree = self.parser.parse(content.encode("utf-8"))
        imports = self._extract_imports(tree.root_node, content)
        patterns: list[ExtractedPattern] = []

        # Walk the AST and extract patterns
        for node in self._walk_tree(tree.root_node):
            pattern = self._try_extract_pattern(
                node, content, source_file, imports, pattern_types
            )
            if pattern:
                patterns.append(pattern)

        return patterns

    def _walk_tree(self, node: Node) -> Iterator[Node]:
        """Walk AST tree depth-first."""
        yield node
        for child in node.children:
            yield from self._walk_tree(child)

    def _extract_imports(self, root: Node, content: str) -> list[str]:
        """Extract import statements from the file."""
        imports: list[str] = []

        for child in root.children:
            if child.type == "import_statement":
                import_text = content[child.start_byte : child.end_byte]
                imports.append(import_text)

        return imports

    def _try_extract_pattern(
        self,
        node: Node,
        content: str,
        source_file: str,
        imports: list[str],
        pattern_types: list[str] | None,
    ) -> ExtractedPattern | None:
        """Try to extract a pattern from a node.

        Returns None if node doesn't match any pattern type.
        """
        # Check for function/method
        if node.type in ("function_declaration", "method_definition"):
            return self._extract_function(node, content, source_file, imports)

        # Check for arrow function (exported or assigned)
        if node.type == "lexical_declaration":
            return self._extract_arrow_function(node, content, source_file, imports)

        # Check for class
        if node.type == "class_declaration":
            return self._extract_class(node, content, source_file, imports)

        # Check for export statement containing function/class
        if node.type == "export_statement":
            return self._extract_from_export(node, content, source_file, imports)

        return None

    def _extract_function(
        self,
        node: Node,
        content: str,
        source_file: str,
        imports: list[str],
    ) -> ExtractedPattern | None:
        """Extract a function declaration."""
        try:
            # Get function name
            name_node = node.child_by_field_name("name")
            if not name_node:
                return None

            name = content[name_node.start_byte : name_node.end_byte]
            full_code = content[node.start_byte : node.end_byte]

            # Find body
            body_node = node.child_by_field_name("body")
            if not body_node:
                return None

            # Split into signature and body
            signature_end = body_node.start_byte - node.start_byte
            if signature_end < 0 or signature_end > len(full_code):
                return None
            signature = full_code[:signature_end].strip()
            body = full_code[signature_end:].strip()
        except (IndexError, ValueError):
            return None

        # Check for React component (starts with capital letter, returns JSX)
        pattern_type = "function"
        if name[0].isupper() and self._contains_jsx(body):
            pattern_type = "component"
        elif name.startswith("use"):
            pattern_type = "hook"

        # Filter related imports
        related_imports = self._filter_related_imports(imports, full_code)

        return ExtractedPattern(
            source_file=source_file,
            pattern_type=pattern_type,
            name=name,
            full_code=full_code,
            signature=signature,
            body=body,
            imports=related_imports,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            language=self.language,
        )

    def _extract_arrow_function(
        self,
        node: Node,
        content: str,
        source_file: str,
        imports: list[str],
    ) -> ExtractedPattern | None:
        """Extract an arrow function from variable declaration."""
        try:
            # Find variable declarator
            for child in node.children:
                if child.type == "variable_declarator":
                    name_node = child.child_by_field_name("name")
                    value_node = child.child_by_field_name("value")

                    if not name_node or not value_node:
                        continue

                    if value_node.type != "arrow_function":
                        # Check for XState machine
                        if self._is_xstate_machine(value_node, content):
                            return self._extract_machine(
                                node, name_node, content, source_file, imports
                            )
                        continue

                    name = content[name_node.start_byte : name_node.end_byte]
                    if not name:
                        continue
                    full_code = content[node.start_byte : node.end_byte]

                    # Find body of arrow function
                    body_node = value_node.child_by_field_name("body")
                    if not body_node:
                        continue

                    # Calculate signature (everything before body)
                    body_start = body_node.start_byte - node.start_byte
                    if body_start < 0 or body_start > len(full_code):
                        continue
                    signature = full_code[:body_start].strip()
                    body = full_code[body_start:].strip()

                    # Determine pattern type
                    pattern_type = "function"
                    if name[0].isupper() and self._contains_jsx(body):
                        pattern_type = "component"
                    elif name.startswith("use"):
                        pattern_type = "hook"

                    related_imports = self._filter_related_imports(imports, full_code)

                    return ExtractedPattern(
                        source_file=source_file,
                        pattern_type=pattern_type,
                        name=name,
                        full_code=full_code,
                        signature=signature,
                        body=body,
                        imports=related_imports,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language=self.language,
                    )
        except (IndexError, ValueError):
            pass

        return None

    def _extract_class(
        self,
        node: Node,
        content: str,
        source_file: str,
        imports: list[str],
    ) -> ExtractedPattern | None:
        """Extract a class declaration."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None

        name = content[name_node.start_byte : name_node.end_byte]
        full_code = content[node.start_byte : node.end_byte]

        # Find body
        body_node = node.child_by_field_name("body")
        if not body_node:
            return None

        signature_end = body_node.start_byte - node.start_byte
        signature = full_code[:signature_end].strip()
        body = full_code[signature_end:].strip()

        related_imports = self._filter_related_imports(imports, full_code)

        return ExtractedPattern(
            source_file=source_file,
            pattern_type="class",
            name=name,
            full_code=full_code,
            signature=signature,
            body=body,
            imports=related_imports,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            language=self.language,
        )

    def _extract_from_export(
        self,
        node: Node,
        content: str,
        source_file: str,
        imports: list[str],
    ) -> ExtractedPattern | None:
        """Extract pattern from export statement."""
        for child in node.children:
            if child.type == "function_declaration":
                pattern = self._extract_function(child, content, source_file, imports)
                if pattern:
                    # Update full_code to include export
                    pattern.full_code = content[node.start_byte : node.end_byte]
                    pattern.signature = "export " + pattern.signature
                return pattern

            if child.type == "class_declaration":
                pattern = self._extract_class(child, content, source_file, imports)
                if pattern:
                    pattern.full_code = content[node.start_byte : node.end_byte]
                    pattern.signature = "export " + pattern.signature
                return pattern

            if child.type == "lexical_declaration":
                pattern = self._extract_arrow_function(
                    child, content, source_file, imports
                )
                if pattern:
                    pattern.full_code = content[node.start_byte : node.end_byte]
                return pattern

        return None

    def _is_xstate_machine(self, node: Node, content: str) -> bool:
        """Check if node is an XState machine definition."""
        if node.type != "call_expression":
            return False

        code = content[node.start_byte : node.end_byte]
        return "createMachine(" in code or "Machine(" in code

    def _extract_machine(
        self,
        decl_node: Node,
        name_node: Node,
        content: str,
        source_file: str,
        imports: list[str],
    ) -> ExtractedPattern | None:
        """Extract XState machine definition."""
        name = content[name_node.start_byte : name_node.end_byte]
        full_code = content[decl_node.start_byte : decl_node.end_byte]

        # Find the object argument (machine config)
        equal_pos = full_code.find("=")
        if equal_pos < 0:
            return None

        signature = full_code[: equal_pos + 1].strip()
        body = full_code[equal_pos + 1 :].strip()

        related_imports = self._filter_related_imports(imports, full_code)

        return ExtractedPattern(
            source_file=source_file,
            pattern_type="machine",
            name=name,
            full_code=full_code,
            signature=signature,
            body=body,
            imports=related_imports,
            start_line=decl_node.start_point[0] + 1,
            end_line=decl_node.end_point[0] + 1,
            language=self.language,
        )

    def _contains_jsx(self, code: str) -> bool:
        """Check if code contains JSX."""
        # Simple heuristic: look for JSX-like patterns
        jsx_patterns = [
            r"<[A-Z][a-zA-Z]*",  # <Component
            r"<[a-z]+\s",  # <div, <span etc.
            r"return\s*\(",  # return ( - common React pattern
        ]
        return any(re.search(p, code) for p in jsx_patterns)

    def _filter_related_imports(
        self, imports: list[str], code: str
    ) -> list[str]:
        """Filter imports to only those used in the code."""
        related: list[str] = []

        for imp in imports:
            # Extract imported names
            # e.g., "import { foo, bar } from 'module'" -> ["foo", "bar"]
            match = re.search(r"\{([^}]+)\}", imp)
            if match:
                names = [n.strip() for n in match.group(1).split(",")]
                if any(re.search(rf"\b{name}\b", code) for name in names):
                    related.append(imp)
            else:
                # Default import: "import Foo from 'module'"
                match = re.search(r"import\s+(\w+)", imp)
                if match and match.group(1) in code:
                    related.append(imp)

        return related


def patterns_to_training_data(
    patterns: list[ExtractedPattern],
    split_ratio: float = 0.5,
) -> list[dict[str, Any]]:
    """Convert extracted patterns to training data format.

    Args:
        patterns: List of extracted patterns
        split_ratio: Where to split for input/output (0.3-0.7)

    Returns:
        List of training examples in mlx-lm format
    """
    examples: list[dict[str, Any]] = []

    INSTRUCTION_TEMPLATES = [
        "Complete this {pattern_type}:",
        "Implement the body of this {pattern_type}:",
        "Fill in the implementation:",
        "Write the code for this {pattern_type}:",
    ]

    for pattern in patterns:
        # Skip very short patterns
        if len(pattern.body) < 30:
            continue

        # Build input context
        input_parts = []
        if pattern.imports:
            input_parts.append("\n".join(pattern.imports))
        input_parts.append(f"// File: {pattern.source_file}")
        input_parts.append(pattern.signature)

        input_text = "\n\n".join(input_parts)

        # Choose instruction
        template = random.choice(INSTRUCTION_TEMPLATES)
        instruction = template.format(pattern_type=pattern.pattern_type)

        # Create mlx-lm format
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{pattern.body}"
        )

        examples.append({"text": text})

    return examples


def extract_patterns_from_repo(
    repo_path: Path,
    extensions: list[str] | None = None,
    pattern_types: list[str] | None = None,
) -> list[ExtractedPattern]:
    """Extract patterns from all files in a repository.

    Args:
        repo_path: Path to repository root
        extensions: File extensions to process (default: .ts, .tsx)
        pattern_types: Types of patterns to extract

    Returns:
        List of all extracted patterns
    """
    if extensions is None:
        extensions = [".ts", ".tsx"]

    all_patterns: list[ExtractedPattern] = []
    errors = 0

    for ext in extensions:
        for file_path in repo_path.rglob(f"*{ext}"):
            # Skip node_modules, dist, etc.
            if any(part.startswith(".") or part in ("node_modules", "dist", "build")
                   for part in file_path.parts):
                continue

            try:
                language = "tsx" if ext == ".tsx" else "typescript"
                extractor = ASTExtractor(language=language)

                patterns = extractor.extract_from_file(file_path, pattern_types)
                all_patterns.extend(patterns)
            except Exception as e:
                errors += 1
                logger.debug(f"Failed to extract from {file_path}: {e}")

    if errors > 0:
        logger.warning(f"Failed to extract from {errors} files")

    logger.info(f"Extracted {len(all_patterns)} patterns from {repo_path}")
    return all_patterns
