"""Custom exceptions for mochi library."""

from __future__ import annotations

from typing import Any


class MochiError(Exception):
    """Base exception for all mochi errors."""

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        if self.context:
            ctx = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} ({ctx})"
        return self.message


class AdapterError(MochiError):
    """Error related to adapter operations."""

    pass


class AdapterNotFoundError(AdapterError):
    """Adapter not found in registry or file system."""

    def __init__(self, adapter_name: str, searched_paths: list[str] | None = None) -> None:
        context = {"adapter_name": adapter_name}
        if searched_paths:
            context["searched_paths"] = searched_paths
        super().__init__(f"Adapter not found: {adapter_name}", context)
        self.adapter_name = adapter_name
        self.searched_paths = searched_paths or []


class TrainingError(MochiError):
    """Error during adapter training."""

    pass


class InferenceError(MochiError):
    """Error during inference."""

    pass


class ConfigurationError(MochiError):
    """Error in configuration."""

    pass


class LSPError(MochiError):
    """Error in LSP communication."""

    pass


class DataError(MochiError):
    """Error in data processing."""

    pass
