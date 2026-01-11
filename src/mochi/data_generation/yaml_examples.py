"""YAML/Config file training example generation.

P1: 設定ファイル(YAML)学習対応
Generates training examples for YAML configuration files
to help the model understand config patterns.

P3: YAML生成強化
- P3.1: ネスト構造対応
- P3.2: コメント抽出・活用
- P3.3: データセットYAML追加
- P3.4: 配列アイテム補完

Law compliance:
- L-config-patterns: Learn YAML/config file patterns from kiri codebase
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mochi.data_generation.alpaca_converter import AlpacaExample


@dataclass
class YAMLSection:
    """A section of a YAML file for training."""

    key: str
    content: str
    file_path: str
    full_yaml: str
    depth: int = 0  # P3.1: ネストの深さ
    comments: list[str] = field(default_factory=list)  # P3.2: 関連コメント


@dataclass
class YAMLArrayItem:
    """P3.4: An array item in YAML for training."""

    parent_key: str
    item_content: str
    item_index: int
    file_path: str
    context_before: str


class YAMLExampleGenerator:
    """Generate training examples from YAML configuration files.

    Creates examples that teach the model:
    1. YAML syntax and structure
    2. Common configuration patterns
    3. kiri-specific config conventions
    4. Config key completion
    5. Value suggestion based on key
    """

    # Templates for YAML completion
    COMPLETION_TEMPLATES = [
        "Complete the following YAML configuration:",
        "Fill in the YAML config values:",
        "Continue this YAML configuration file:",
        "Write the configuration for this YAML section:",
    ]

    # Templates for key completion
    KEY_TEMPLATES = [
        "What YAML key should come next?",
        "Complete the YAML key name:",
        "Suggest the next configuration key:",
    ]

    # Templates for value completion
    VALUE_TEMPLATES = [
        "What value should '{key}' have in this config?",
        "Complete the value for '{key}':",
        "Suggest a value for the '{key}' configuration:",
    ]

    # Templates for structure explanation
    STRUCTURE_TEMPLATES = [
        "Explain the structure of this YAML configuration:",
        "What does this configuration define?",
        "Describe the purpose of this config section:",
    ]

    # P3.2: Templates using comment context
    COMMENT_CONTEXT_TEMPLATES = [
        "Complete the YAML value. {comment}",
        "Fill in the config. Context: {comment}",
        "What value should this have? Hint: {comment}",
    ]

    # P3.3: Templates for dataset YAML
    DATASET_TEMPLATES = [
        "Complete this query definition in the dataset:",
        "Fill in the query metadata:",
        "Continue this evaluation dataset entry:",
        "Complete the expected results section:",
    ]

    # P3.4: Templates for array items
    ARRAY_ITEM_TEMPLATES = [
        "Add the next array item:",
        "Complete this list entry:",
        "What should the next item be?",
        "Continue the list with a new entry:",
    ]

    def __init__(
        self,
        project_name: str = "kiri",
        config_conventions: dict[str, Any] | None = None,
    ) -> None:
        """Initialize generator.

        Args:
            project_name: Project name for context
            config_conventions: Optional conventions for config patterns
        """
        self.project_name = project_name
        self.conventions = config_conventions or {}

    def load_yaml_files(
        self,
        directory: Path,
        extensions: tuple[str, ...] = (".yaml", ".yml"),
    ) -> list[YAMLSection]:
        """Load YAML files from a directory.

        Args:
            directory: Directory to search
            extensions: File extensions to include

        Returns:
            List of YAMLSection objects
        """
        sections = []

        for ext in extensions:
            for yaml_file in directory.rglob(f"*{ext}"):
                try:
                    content = yaml_file.read_text(encoding="utf-8")
                    # Parse to validate
                    yaml.safe_load(content)

                    # Split into sections by top-level keys
                    file_sections = self._split_yaml_sections(
                        content, str(yaml_file.relative_to(directory))
                    )
                    sections.extend(file_sections)
                except Exception:
                    # Skip invalid YAML files
                    continue

        return sections

    def _split_yaml_sections(
        self,
        content: str,
        file_path: str,
    ) -> list[YAMLSection]:
        """Split YAML content into sections by top-level keys."""
        sections = []
        lines = content.split("\n")

        current_key = None
        current_lines = []
        indent_level = 0

        for line in lines:
            stripped = line.strip()

            # Skip comments and empty lines for key detection
            if not stripped or stripped.startswith("#"):
                current_lines.append(line)
                continue

            # Check if this is a top-level key (no indentation)
            if not line.startswith(" ") and not line.startswith("\t"):
                if ":" in line:
                    # Save previous section
                    if current_key and current_lines:
                        sections.append(
                            YAMLSection(
                                key=current_key,
                                content="\n".join(current_lines).strip(),
                                file_path=file_path,
                                full_yaml=content,
                            )
                        )

                    # Start new section
                    current_key = line.split(":")[0].strip()
                    current_lines = [line]
                else:
                    current_lines.append(line)
            else:
                current_lines.append(line)

        # Don't forget the last section
        if current_key and current_lines:
            sections.append(
                YAMLSection(
                    key=current_key,
                    content="\n".join(current_lines).strip(),
                    file_path=file_path,
                    full_yaml=content,
                )
            )

        return sections

    def generate_completion_examples(
        self,
        sections: list[YAMLSection],
        num_examples: int | None = None,
    ) -> list[AlpacaExample]:
        """Generate YAML completion examples.

        Args:
            sections: YAML sections to use
            num_examples: Max examples to generate (None = all)

        Returns:
            List of AlpacaExample for training
        """
        examples = []

        for section in sections:
            lines = section.content.split("\n")
            if len(lines) < 2:
                continue

            # Create completion at different points
            split_points = [
                max(1, len(lines) // 3),
                len(lines) // 2,
                min(len(lines) - 1, len(lines) * 2 // 3),
            ]

            for split_point in split_points:
                if split_point < 1:
                    continue

                context_lines = lines[:split_point]
                completion_lines = lines[split_point:]

                if not completion_lines:
                    continue

                # Build context with file info
                context = f"# File: {section.file_path}\n\n" + "\n".join(context_lines)
                completion = "\n".join(completion_lines)

                template = random.choice(self.COMPLETION_TEMPLATES)
                examples.append(
                    AlpacaExample(
                        instruction=template,
                        input=context,
                        output=completion,
                    )
                )

        if num_examples:
            random.shuffle(examples)
            examples = examples[:num_examples]

        return examples

    def generate_key_completion_examples(
        self,
        sections: list[YAMLSection],
        num_examples: int = 50,
    ) -> list[AlpacaExample]:
        """Generate examples for completing YAML keys.

        Args:
            sections: YAML sections to use
            num_examples: Number of examples to generate

        Returns:
            List of AlpacaExample for training
        """
        examples = []

        for section in sections:
            lines = section.content.split("\n")

            for i, line in enumerate(lines):
                if ":" not in line:
                    continue

                # Get the key
                key_part = line.split(":")[0]
                key = key_part.strip().lstrip("- ")

                if not key or key.startswith("#"):
                    continue

                # Context: lines before, with partial key
                context_lines = lines[:i]
                # Show just the indentation
                indent = len(key_part) - len(key_part.lstrip())
                partial = key_part[:indent]

                if context_lines:
                    context = "\n".join(context_lines) + "\n" + partial
                else:
                    context = partial

                # Add file context
                context = f"# File: {section.file_path}\n\n" + context

                template = random.choice(self.KEY_TEMPLATES)
                examples.append(
                    AlpacaExample(
                        instruction=template,
                        input=context,
                        output=key + ":",
                    )
                )

        random.shuffle(examples)
        return examples[:num_examples]

    def generate_value_completion_examples(
        self,
        sections: list[YAMLSection],
        num_examples: int = 50,
    ) -> list[AlpacaExample]:
        """Generate examples for completing YAML values.

        Args:
            sections: YAML sections to use
            num_examples: Number of examples to generate

        Returns:
            List of AlpacaExample for training
        """
        examples = []

        for section in sections:
            lines = section.content.split("\n")

            for i, line in enumerate(lines):
                if ":" not in line:
                    continue

                parts = line.split(":", 1)
                if len(parts) < 2:
                    continue

                key_part = parts[0]
                value_part = parts[1].strip()

                key = key_part.strip().lstrip("- ")

                # Skip empty values or comments
                if not value_part or value_part.startswith("#"):
                    continue

                # Context: lines before + key with colon
                context_lines = lines[:i]
                context_lines.append(key_part + ": ")

                context = "\n".join(context_lines)
                context = f"# File: {section.file_path}\n\n" + context

                template = random.choice(self.VALUE_TEMPLATES)
                instruction = template.format(key=key)

                examples.append(
                    AlpacaExample(
                        instruction=instruction,
                        input=context,
                        output=value_part,
                    )
                )

        random.shuffle(examples)
        return examples[:num_examples]

    # ================================================================
    # P3.1: ネスト構造対応
    # ================================================================

    def _split_nested_sections(
        self,
        content: str,
        file_path: str,
        max_depth: int = 2,
    ) -> list[YAMLSection]:
        """P3.1: Split YAML content into nested sections up to max_depth.

        Args:
            content: YAML content
            file_path: File path for context
            max_depth: Maximum nesting depth to extract (default: 2)

        Returns:
            List of YAMLSection objects including nested sections
        """
        sections = []
        lines = content.split("\n")

        # Track nested structure
        current_path: list[tuple[str, int]] = []  # (key, indent)
        section_start: dict[str, int] = {}  # key -> start line index
        section_lines: dict[str, list[str]] = {}  # key -> lines
        comments_buffer: list[str] = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Collect comments
            if stripped.startswith("#"):
                comments_buffer.append(stripped[1:].strip())
                continue

            if not stripped:
                continue

            # Calculate indent level
            indent = len(line) - len(line.lstrip())
            indent_level = indent // 2  # Assume 2-space indent

            # Check for key
            if ":" in stripped and not stripped.startswith("-"):
                key = stripped.split(":")[0].strip()

                # Pop keys with same or higher indent from path
                while current_path and current_path[-1][1] >= indent:
                    current_path.pop()

                # Build full path
                path_parts = [k for k, _ in current_path] + [key]
                full_key = ".".join(path_parts)

                # Only track up to max_depth
                if len(path_parts) <= max_depth:
                    current_path.append((key, indent))
                    section_start[full_key] = i
                    section_lines[full_key] = [line]

                    # Create section with comments
                    sections.append(
                        YAMLSection(
                            key=full_key,
                            content=line,
                            file_path=file_path,
                            full_yaml=content,
                            depth=len(path_parts) - 1,
                            comments=comments_buffer.copy(),
                        )
                    )
                    comments_buffer.clear()

        # Update section content with child lines
        for section in sections:
            if section.key in section_start:
                start_idx = section_start[section.key]
                section_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())

                # Collect all lines under this section
                section_content = [lines[start_idx]]
                for j in range(start_idx + 1, len(lines)):
                    line = lines[j]
                    if not line.strip():
                        section_content.append(line)
                        continue

                    line_indent = len(line) - len(line.lstrip())
                    if line_indent > section_indent:
                        section_content.append(line)
                    else:
                        break

                section.content = "\n".join(section_content).strip()

        return sections

    def load_yaml_files_nested(
        self,
        directory: Path,
        max_depth: int = 2,
        extensions: tuple[str, ...] = (".yaml", ".yml"),
    ) -> list[YAMLSection]:
        """P3.1: Load YAML files with nested section extraction.

        Args:
            directory: Directory to search
            max_depth: Maximum nesting depth
            extensions: File extensions to include

        Returns:
            List of YAMLSection objects including nested sections
        """
        sections = []

        for ext in extensions:
            for yaml_file in directory.rglob(f"*{ext}"):
                # Skip lock files
                if "lock" in yaml_file.name.lower():
                    continue

                try:
                    content = yaml_file.read_text(encoding="utf-8")
                    yaml.safe_load(content)

                    nested_sections = self._split_nested_sections(
                        content,
                        str(yaml_file.relative_to(directory)),
                        max_depth,
                    )
                    sections.extend(nested_sections)
                except Exception:
                    continue

        return sections

    # ================================================================
    # P3.2: コメント抽出・活用
    # ================================================================

    def _extract_inline_comment(self, line: str) -> str | None:
        """P3.2: Extract inline comment from a YAML line.

        Args:
            line: YAML line

        Returns:
            Comment text or None
        """
        # Match inline comments: value # comment
        match = re.search(r"#\s*(.+)$", line)
        if match and ":" in line:  # Only for key-value lines
            return match.group(1).strip()
        return None

    def generate_comment_guided_examples(
        self,
        sections: list[YAMLSection],
        num_examples: int = 50,
    ) -> list[AlpacaExample]:
        """P3.2: Generate examples using comments as training signals.

        Args:
            sections: YAML sections
            num_examples: Number of examples

        Returns:
            List of AlpacaExample
        """
        examples = []

        for section in sections:
            lines = section.content.split("\n")

            for i, line in enumerate(lines):
                comment = self._extract_inline_comment(line)
                if not comment:
                    continue

                # Extract key and value
                if ":" not in line:
                    continue

                parts = line.split(":", 1)
                key_part = parts[0]
                value_part = parts[1].split("#")[0].strip() if len(parts) > 1 else ""

                if not value_part:
                    continue

                # Context: lines before + key with colon
                context_lines = lines[:i]
                context_lines.append(key_part + ": ")
                context = "\n".join(context_lines)
                context = f"# File: {section.file_path}\n\n" + context

                template = random.choice(self.COMMENT_CONTEXT_TEMPLATES)
                instruction = template.format(comment=comment)

                examples.append(
                    AlpacaExample(
                        instruction=instruction,
                        input=context,
                        output=value_part,
                    )
                )

        random.shuffle(examples)
        return examples[:num_examples]

    # ================================================================
    # P3.3: データセットYAML追加
    # ================================================================

    def load_dataset_yaml_files(
        self,
        datasets_dir: Path,
    ) -> list[YAMLSection]:
        """P3.3: Load YAML files from datasets directory.

        Args:
            datasets_dir: Path to datasets directory

        Returns:
            List of YAMLSection for dataset files
        """
        sections = []

        if not datasets_dir.exists():
            return sections

        for yaml_file in datasets_dir.glob("*.yaml"):
            try:
                content = yaml_file.read_text(encoding="utf-8")
                data = yaml.safe_load(content)

                if not isinstance(data, dict):
                    continue

                # Extract queries section if present
                if "queries" in data and isinstance(data["queries"], list):
                    queries = data["queries"]
                    for i, query in enumerate(queries):
                        if isinstance(query, dict):
                            query_yaml = yaml.dump(
                                query, default_flow_style=False, allow_unicode=True
                            )
                            sections.append(
                                YAMLSection(
                                    key=f"query-{i}",
                                    content=query_yaml,
                                    file_path=str(yaml_file.name),
                                    full_yaml=content,
                                    depth=1,
                                )
                            )

                # Also extract metadata sections
                for key in ["defaultParams", "schemaVersion", "name", "description"]:
                    if key in data:
                        value = data[key]
                        section_yaml = yaml.dump(
                            {key: value}, default_flow_style=False, allow_unicode=True
                        )
                        sections.append(
                            YAMLSection(
                                key=key,
                                content=section_yaml.strip(),
                                file_path=str(yaml_file.name),
                                full_yaml=content,
                            )
                        )

            except Exception:
                continue

        return sections

    def generate_dataset_examples(
        self,
        sections: list[YAMLSection],
        num_examples: int = 100,
    ) -> list[AlpacaExample]:
        """P3.3: Generate training examples from dataset YAML.

        Args:
            sections: Dataset YAML sections
            num_examples: Number of examples

        Returns:
            List of AlpacaExample
        """
        examples = []

        for section in sections:
            if not section.content:
                continue

            lines = section.content.split("\n")
            if len(lines) < 2:
                continue

            # Create completions at different points
            for split_ratio in [0.3, 0.5, 0.7]:
                split_point = max(1, int(len(lines) * split_ratio))

                context_lines = lines[:split_point]
                completion_lines = lines[split_point:]

                if not completion_lines:
                    continue

                context = f"# File: datasets/{section.file_path}\n\n" + "\n".join(
                    context_lines
                )
                completion = "\n".join(completion_lines)

                template = random.choice(self.DATASET_TEMPLATES)
                examples.append(
                    AlpacaExample(
                        instruction=template,
                        input=context,
                        output=completion,
                    )
                )

        random.shuffle(examples)
        return examples[:num_examples]

    # ================================================================
    # P3.4: 配列アイテム補完
    # ================================================================

    def _extract_array_items(
        self,
        content: str,
        file_path: str,
    ) -> list[YAMLArrayItem]:
        """P3.4: Extract array items from YAML content.

        Args:
            content: YAML content
            file_path: File path

        Returns:
            List of YAMLArrayItem
        """
        items = []
        lines = content.split("\n")

        current_array_key = None
        current_item_lines: list[str] = []
        item_start_indent = 0
        item_index = 0
        context_before = ""

        for i, line in enumerate(lines):
            stripped = line.strip()

            if not stripped:
                continue

            # Check for array key (key followed by array)
            if ":" in stripped and not stripped.startswith("-"):
                key = stripped.split(":")[0].strip()
                # Check if next non-empty line is array item
                for j in range(i + 1, min(i + 5, len(lines))):
                    next_stripped = lines[j].strip()
                    if next_stripped.startswith("-"):
                        current_array_key = key
                        item_index = 0
                        context_before = "\n".join(lines[: i + 1])
                        break
                    elif next_stripped and not next_stripped.startswith("#"):
                        break

            # Check for array item
            if stripped.startswith("-"):
                indent = len(line) - len(line.lstrip())

                # If we're in an array and this is a new item
                if current_array_key:
                    if current_item_lines:
                        # Save previous item
                        items.append(
                            YAMLArrayItem(
                                parent_key=current_array_key,
                                item_content="\n".join(current_item_lines),
                                item_index=item_index,
                                file_path=file_path,
                                context_before=context_before,
                            )
                        )
                        item_index += 1

                    current_item_lines = [line]
                    item_start_indent = indent

            elif current_item_lines:
                # Continue current item if indented
                indent = len(line) - len(line.lstrip())
                if indent > item_start_indent:
                    current_item_lines.append(line)
                else:
                    # End of item
                    items.append(
                        YAMLArrayItem(
                            parent_key=current_array_key,
                            item_content="\n".join(current_item_lines),
                            item_index=item_index,
                            file_path=file_path,
                            context_before=context_before,
                        )
                    )
                    item_index += 1
                    current_item_lines = []
                    current_array_key = None

        # Don't forget last item
        if current_item_lines and current_array_key:
            items.append(
                YAMLArrayItem(
                    parent_key=current_array_key,
                    item_content="\n".join(current_item_lines),
                    item_index=item_index,
                    file_path=file_path,
                    context_before=context_before,
                )
            )

        return items

    def generate_array_item_examples(
        self,
        sections: list[YAMLSection],
        num_examples: int = 50,
    ) -> list[AlpacaExample]:
        """P3.4: Generate examples for array item completion.

        Args:
            sections: YAML sections
            num_examples: Number of examples

        Returns:
            List of AlpacaExample
        """
        examples = []

        for section in sections:
            items = self._extract_array_items(section.content, section.file_path)

            for i, item in enumerate(items):
                if i == 0:
                    continue  # Skip first item, no context

                # Context: previous items
                prev_items = items[:i]
                context_parts = [f"# File: {section.file_path}\n"]
                context_parts.append(item.context_before)
                for prev in prev_items[-3:]:  # Last 3 items for context
                    context_parts.append(prev.item_content)

                context = "\n".join(context_parts)

                template = random.choice(self.ARRAY_ITEM_TEMPLATES)
                examples.append(
                    AlpacaExample(
                        instruction=template,
                        input=context,
                        output=item.item_content,
                    )
                )

        random.shuffle(examples)
        return examples[:num_examples]

    def generate_all(
        self,
        sections: list[YAMLSection],
        completion_count: int = 100,
        key_count: int = 50,
        value_count: int = 50,
        comment_count: int = 50,  # P3.2
        array_count: int = 50,  # P3.4
    ) -> list[AlpacaExample]:
        """Generate all types of YAML examples.

        Args:
            sections: YAML sections to use
            completion_count: Number of completion examples
            key_count: Number of key completion examples
            value_count: Number of value completion examples
            comment_count: Number of comment-guided examples (P3.2)
            array_count: Number of array item examples (P3.4)

        Returns:
            Combined list of all example types
        """
        all_examples = []

        all_examples.extend(
            self.generate_completion_examples(sections, completion_count)
        )
        all_examples.extend(
            self.generate_key_completion_examples(sections, key_count)
        )
        all_examples.extend(
            self.generate_value_completion_examples(sections, value_count)
        )
        # P3.2: コメント活用例
        all_examples.extend(
            self.generate_comment_guided_examples(sections, comment_count)
        )
        # P3.4: 配列アイテム補完例
        all_examples.extend(
            self.generate_array_item_examples(sections, array_count)
        )

        random.shuffle(all_examples)
        return all_examples


def generate_yaml_training_data(
    yaml_dir: Path,
    project_name: str = "kiri",
    completion_count: int = 100,
    key_count: int = 50,
    value_count: int = 50,
    comment_count: int = 50,  # P3.2
    array_count: int = 50,  # P3.4
    dataset_count: int = 100,  # P3.3
    use_nested: bool = True,  # P3.1
) -> list[AlpacaExample]:
    """Convenience function to generate YAML training data.

    Args:
        yaml_dir: Directory containing YAML files
        project_name: Project name
        completion_count: Number of completion examples
        key_count: Number of key completion examples
        value_count: Number of value completion examples
        comment_count: Number of comment-guided examples (P3.2)
        array_count: Number of array item examples (P3.4)
        dataset_count: Number of dataset YAML examples (P3.3)
        use_nested: Use nested section extraction (P3.1)

    Returns:
        List of AlpacaExample for training
    """
    generator = YAMLExampleGenerator(project_name=project_name)

    # P3.1: Use nested extraction if enabled
    if use_nested:
        sections = generator.load_yaml_files_nested(yaml_dir, max_depth=2)
    else:
        sections = generator.load_yaml_files(yaml_dir)

    if not sections:
        return []

    all_examples = generator.generate_all(
        sections,
        completion_count=completion_count,
        key_count=key_count,
        value_count=value_count,
        comment_count=comment_count,
        array_count=array_count,
    )

    # P3.3: Add dataset YAML examples
    datasets_dir = yaml_dir / "datasets"
    if datasets_dir.exists():
        dataset_sections = generator.load_dataset_yaml_files(datasets_dir)
        if dataset_sections:
            dataset_examples = generator.generate_dataset_examples(
                dataset_sections, dataset_count
            )
            all_examples.extend(dataset_examples)

    random.shuffle(all_examples)
    return all_examples
