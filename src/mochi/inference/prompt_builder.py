"""Prompt builder for inference.

Constructs prompts in Alpaca format for code completion tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CompletionContext:
    """Context for a code completion request."""

    instruction: str
    input_code: str
    file_path: str | None = None
    lsp_context: str | None = None
    additional_context: str | None = None


class PromptBuilder:
    """Build prompts for code completion.

    Supports Alpaca format with optional LSP context injection.

    Usage:
        builder = PromptBuilder()
        prompt = builder.build_completion_prompt(
            instruction="Fill in the code",
            input_code="const users = await db.",
            file_path="src/api/users.ts",
            lsp_context="// Available methods: all<T>(), get<T>(), run()",
        )
    """

    # Alpaca format template
    ALPACA_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

    ALPACA_TEMPLATE_NO_INPUT = """### Instruction:
{instruction}

### Response:
"""

    def __init__(
        self,
        include_file_path: bool = True,
        include_lsp_context: bool = True,
        max_context_lines: int = 50,
    ) -> None:
        """Initialize prompt builder.

        Args:
            include_file_path: Include file path in input
            include_lsp_context: Include LSP context in input
            max_context_lines: Maximum lines of context to include
        """
        self.include_file_path = include_file_path
        self.include_lsp_context = include_lsp_context
        self.max_context_lines = max_context_lines

    def build_completion_prompt(
        self,
        instruction: str,
        input_code: str,
        file_path: str | None = None,
        lsp_context: str | None = None,
        additional_context: str | None = None,
    ) -> str:
        """Build a completion prompt.

        Args:
            instruction: Instruction for the model
            input_code: Code to complete
            file_path: Optional file path for context
            lsp_context: Optional LSP context (types, methods)
            additional_context: Additional context to include

        Returns:
            Formatted prompt string
        """
        # Build input section
        input_parts = []

        # Add file path
        if file_path and self.include_file_path:
            input_parts.append(f"// File: {file_path}")

        # Add LSP context
        if lsp_context and self.include_lsp_context:
            # Truncate if too long
            context_lines = lsp_context.split("\n")
            if len(context_lines) > self.max_context_lines:
                context_lines = context_lines[: self.max_context_lines]
                context_lines.append("// ... (truncated)")
            input_parts.append("\n".join(context_lines))

        # Add additional context
        if additional_context:
            input_parts.append(additional_context)

        # Add the input code
        input_parts.append(input_code)

        # Join all parts
        full_input = "\n".join(input_parts)

        # Format with template
        if full_input.strip():
            return self.ALPACA_TEMPLATE.format(
                instruction=instruction,
                input=full_input,
            )
        else:
            return self.ALPACA_TEMPLATE_NO_INPUT.format(
                instruction=instruction,
            )

    def build_from_context(self, context: CompletionContext) -> str:
        """Build prompt from CompletionContext.

        Args:
            context: Completion context

        Returns:
            Formatted prompt string
        """
        return self.build_completion_prompt(
            instruction=context.instruction,
            input_code=context.input_code,
            file_path=context.file_path,
            lsp_context=context.lsp_context,
            additional_context=context.additional_context,
        )

    def parse_response(self, response: str) -> str:
        """Parse model response to extract the completion.

        Args:
            response: Raw model response

        Returns:
            Extracted completion text
        """
        # If response contains the template markers, extract just the response part
        if "### Response:" in response:
            parts = response.split("### Response:")
            if len(parts) > 1:
                return parts[-1].strip()

        return response.strip()
