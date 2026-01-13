"""Inference engine for code completion.

Provides a high-level API for running inference with mochi adapters.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.async_utils import run_sync
from ..core.exceptions import AdapterError, InferenceError
from ..core.types import InferenceConfig
from .prompt_builder import CompletionContext, PromptBuilder

if TYPE_CHECKING:
    from ..adapters.adapter_stack import AdapterStack
    from ..adapters.base_adapter import BaseAdapter
    from ..adapters.project_adapter import ProjectAdapter
    from ..lsp.client import LSPClient
    from ..lsp.context_extractor import ContextExtractor

logger = logging.getLogger(__name__)


class InferenceEngine:
    """High-level inference engine for code completion.

    InferenceEngine combines adapter management, LSP context extraction,
    and prompt building to provide a simple API for code completion.

    Usage:
        # With adapter stack
        engine = InferenceEngine(adapter_stack=stack)

        # Or with single adapter
        engine = InferenceEngine(adapter=project_adapter)

        # With LSP context
        engine = InferenceEngine(
            adapter_stack=stack,
            lsp_client=lsp_client,
        )

        # Complete code
        result = engine.complete(
            instruction="Fill in the code",
            input_code="const users = await db.",
            file_path="src/api/users.ts",
        )
    """

    def __init__(
        self,
        adapter: BaseAdapter | ProjectAdapter | None = None,
        adapter_stack: AdapterStack | None = None,
        lsp_client: LSPClient | None = None,
        context_extractor: ContextExtractor | None = None,
        config: InferenceConfig | None = None,
    ) -> None:
        """Initialize inference engine.

        Args:
            adapter: Single adapter for inference (mutually exclusive with adapter_stack)
            adapter_stack: Stack of adapters for inference
            lsp_client: Optional LSP client for context extraction (legacy)
            context_extractor: Optional ContextExtractor for enhanced context extraction
            config: Inference configuration
        """
        if adapter is None and adapter_stack is None:
            raise InferenceError("Either adapter or adapter_stack must be provided")
        if adapter is not None and adapter_stack is not None:
            raise InferenceError("Cannot use both adapter and adapter_stack")

        self._adapter = adapter
        self._adapter_stack = adapter_stack
        self._lsp_client = lsp_client
        self._context_extractor = context_extractor
        self.config = config or InferenceConfig()
        self._prompt_builder = PromptBuilder(
            include_file_path=True,
            include_lsp_context=self.config.use_lsp_context,
            max_context_lines=self.config.lsp_context_lines,
        )

    @property
    def active_adapter(self) -> BaseAdapter | ProjectAdapter:
        """Get the active adapter."""
        if self._adapter_stack:
            return self._adapter_stack.primary
        return self._adapter  # type: ignore

    def complete(
        self,
        instruction: str,
        input_code: str,
        file_path: str | None = None,
        max_tokens: int | None = None,
        min_tokens: int | None = None,
        temperature: float | None = None,
        repetition_penalty: float | None = None,
        use_lsp: bool | None = None,
    ) -> str:
        """Complete code based on instruction and input.

        Args:
            instruction: What to do (e.g., "Fill in the code")
            input_code: Code to complete
            file_path: Optional file path for context
            max_tokens: Max tokens to generate (overrides config)
            min_tokens: Min tokens before allowing EOS (prevents short outputs)
            temperature: Sampling temperature (overrides config)
            repetition_penalty: Repetition penalty (overrides config)
            use_lsp: Whether to use LSP context (overrides config)

        Returns:
            Generated completion

        Raises:
            InferenceError: If inference fails
        """
        # Get LSP context if enabled
        lsp_context = None
        use_lsp_context = use_lsp if use_lsp is not None else self.config.use_lsp_context

        if use_lsp_context and self._lsp_client and file_path:
            try:
                lsp_context = self._get_lsp_context(file_path, input_code)
            except Exception as e:
                logger.warning(f"Failed to get LSP context: {e}")

        # Build prompt
        context = CompletionContext(
            instruction=instruction,
            input_code=input_code,
            file_path=file_path,
            lsp_context=lsp_context,
        )
        prompt = self._prompt_builder.build_from_context(context)

        # Run inference
        try:
            if self._adapter_stack:
                response = self._adapter_stack.generate(
                    prompt=prompt,
                    max_tokens=max_tokens or self.config.max_tokens,
                    min_tokens=min_tokens or 0,
                    temperature=temperature or self.config.temperature,
                    top_p=self.config.top_p,
                    repetition_penalty=repetition_penalty or self.config.repetition_penalty,
                )
            else:
                response = self._adapter.generate(
                    prompt=prompt,
                    max_tokens=max_tokens or self.config.max_tokens,
                    min_tokens=min_tokens or 0,
                    temperature=temperature or self.config.temperature,
                    top_p=self.config.top_p,
                    repetition_penalty=repetition_penalty or self.config.repetition_penalty,
                )

            # Parse response
            completion = self._prompt_builder.parse_response(response)
            return completion

        except AdapterError:
            raise
        except Exception as e:
            raise InferenceError(
                "Inference failed",
                {"error": str(e), "instruction": instruction},
            ) from e

    def _get_lsp_context(self, file_path: str, input_code: str) -> str | None:
        """Get LSP context for the given file and code.

        Uses ContextExtractor for enhanced context extraction including:
        - Receiver type detection (e.g., "db." -> DuckDBClient)
        - Method signatures with parameters and return types
        - Filtered completions (no global builtins)

        Args:
            file_path: Path to the file
            input_code: Current code at cursor

        Returns:
            LSP context string or None
        """
        # Prefer ContextExtractor if available
        if self._context_extractor:
            return self._get_context_via_extractor(file_path, input_code)

        # Legacy fallback: direct LSP client usage
        if self._lsp_client:
            return self._get_context_via_lsp_client(file_path, input_code)

        return None

    def _get_context_via_extractor(self, file_path: str, input_code: str) -> str | None:
        """Extract context using ContextExtractor (async wrapper).

        Args:
            file_path: Path to the file
            input_code: Current code at cursor

        Returns:
            Formatted context string or None
        """
        try:
            # Detect cursor position from input_code
            line, character = self._detect_cursor_position(input_code)

            # Run async extraction in sync context
            context_block = run_sync(
                self._context_extractor.extract_at_position(
                    Path(file_path),
                    line,
                    character,
                    include_schema=True,
                )
            )

            # Return formatted context if not empty
            if context_block and not context_block.is_empty():
                return context_block.format(detailed=True)

            return None

        except Exception as e:
            logger.debug(f"ContextExtractor extraction failed: {e}")
            return None

    def _get_context_via_lsp_client(self, file_path: str, input_code: str) -> str | None:
        """Legacy context extraction using direct LSP client.

        Args:
            file_path: Path to the file
            input_code: Current code at cursor

        Returns:
            Context string or None
        """
        try:
            context_parts = []

            # Get completions (available methods/properties)
            completions = self._lsp_client.get_completions(file_path, input_code)
            if completions:
                context_parts.append("// Available completions:")
                for item in completions[: self.config.lsp_context_lines]:
                    if hasattr(item, "label") and hasattr(item, "detail"):
                        context_parts.append(f"//   {item.label}: {item.detail}")
                    elif isinstance(item, str):
                        context_parts.append(f"//   {item}")

            return "\n".join(context_parts) if context_parts else None

        except Exception as e:
            logger.debug(f"LSP context extraction failed: {e}")
            return None

    def _detect_cursor_position(self, input_code: str) -> tuple[int, int]:
        """Detect cursor position from input code.

        Looks for common completion trigger points:
        - After a dot (method/property access): "obj."
        - After opening parenthesis (arguments): "func("
        - End of code if no specific trigger found

        Args:
            input_code: Code string to analyze

        Returns:
            Tuple of (line_number, character_position), 0-indexed
        """
        lines = input_code.split("\n")

        # Find the last meaningful position
        for line_idx in range(len(lines) - 1, -1, -1):
            line = lines[line_idx]

            # Skip empty lines
            if not line.strip():
                continue

            # Look for trigger characters from the end
            for char_idx in range(len(line) - 1, -1, -1):
                char = line[char_idx]

                # After a dot: "db." -> position after the dot
                if char == ".":
                    return (line_idx, char_idx + 1)

                # After opening paren: "func(" -> position inside
                if char == "(":
                    return (line_idx, char_idx + 1)

            # If no trigger found on this line, return end of line
            return (line_idx, len(line))

        # Default: start of first line
        return (0, 0)

    def set_context_extractor(self, context_extractor: ContextExtractor) -> None:
        """Set or update the ContextExtractor.

        Args:
            context_extractor: ContextExtractor instance
        """
        self._context_extractor = context_extractor

    def batch_complete(
        self,
        items: list[dict[str, Any]],
        max_tokens: int | None = None,
    ) -> list[str]:
        """Complete multiple items in batch.

        Args:
            items: List of dicts with 'instruction', 'input_code', 'file_path' keys
            max_tokens: Max tokens per completion

        Returns:
            List of completions
        """
        results = []
        for item in items:
            try:
                result = self.complete(
                    instruction=item["instruction"],
                    input_code=item["input_code"],
                    file_path=item.get("file_path"),
                    max_tokens=max_tokens,
                )
                results.append(result)
            except InferenceError as e:
                logger.error(f"Batch item failed: {e}")
                results.append("")

        return results

    def set_lsp_client(self, lsp_client: LSPClient) -> None:
        """Set or update the LSP client.

        Args:
            lsp_client: LSP client instance
        """
        self._lsp_client = lsp_client

    def set_adapter(
        self,
        adapter: BaseAdapter | ProjectAdapter | None = None,
        adapter_stack: AdapterStack | None = None,
    ) -> None:
        """Update the adapter or adapter stack.

        Args:
            adapter: Single adapter for inference
            adapter_stack: Stack of adapters for inference
        """
        if adapter is None and adapter_stack is None:
            raise InferenceError("Either adapter or adapter_stack must be provided")
        if adapter is not None and adapter_stack is not None:
            raise InferenceError("Cannot use both adapter and adapter_stack")

        self._adapter = adapter
        self._adapter_stack = adapter_stack

    def __repr__(self) -> str:
        if self._adapter_stack:
            return f"InferenceEngine(adapter_stack={self._adapter_stack})"
        return f"InferenceEngine(adapter={self._adapter})"
