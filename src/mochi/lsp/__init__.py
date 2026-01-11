"""LSP-based context extraction for training data.

This module provides utilities to extract type information, available methods,
and other context from source code using Language Server Protocol (LSP).

Terms:
- LSPClient: Manages communication with language server (tsserver, pylsp, etc.)
- ContextExtractor: Extracts and formats context blocks for training data
- ContextBlock: Formatted context information for training examples

Laws:
- L-fallback-graceful: LSP failures don't break training data generation
- L-context-format: Context uses "// Available methods: ..." format
- L-batch-efficiency: Single file processing < 10 seconds
"""

from mochi.lsp.client import CompletionItem, HoverInfo, LSPClient, SymbolInfo
from mochi.lsp.context_extractor import (
    ContextBlock,
    ContextExtractor,
    MethodSignature,
    SchemaInfo,
    TypeInfo,
    create_context_extractor,
    extract_batch_context,
)

__all__ = [
    "LSPClient",
    "CompletionItem",
    "HoverInfo",
    "SymbolInfo",
    "ContextExtractor",
    "ContextBlock",
    "MethodSignature",
    "TypeInfo",
    "SchemaInfo",
    "create_context_extractor",
    "extract_batch_context",
]
