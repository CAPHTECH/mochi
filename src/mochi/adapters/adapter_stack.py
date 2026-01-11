"""AdapterStack for composing multiple adapters at runtime.

AdapterStack allows combining multiple adapters with configurable weights,
enabling a layered approach where common patterns (BaseAdapter) are combined
with project-specific knowledge (ProjectAdapter).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.exceptions import AdapterError
from ..core.types import AdapterStackConfig

if TYPE_CHECKING:
    from .base_adapter import BaseAdapter
    from .project_adapter import ProjectAdapter

logger = logging.getLogger(__name__)


@dataclass
class StackedAdapter:
    """An adapter with its weight in the stack."""

    adapter: BaseAdapter | ProjectAdapter
    weight: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"Weight must be between 0 and 1, got {self.weight}")


class AdapterStack:
    """Stack of multiple adapters for combined inference.

    AdapterStack combines multiple adapters with configurable weights.
    During inference, the outputs are blended according to the weights.

    Note: Currently, MLX-lm doesn't support true adapter stacking at inference.
    This implementation uses the highest-weighted adapter for generation,
    but the architecture supports future enhancement when MLX-lm adds stacking.

    Usage:
        # Create stack with base (30%) and project (70%) adapters
        stack = AdapterStack([
            (base_adapter, 0.3),
            (project_adapter, 0.7),
        ])

        # Generate using the stack
        result = stack.generate(prompt, max_tokens=256)

        # Access individual adapters
        for adapter, weight in stack.adapters:
            print(f"{adapter.name}: {weight:.0%}")
    """

    def __init__(
        self,
        adapters: list[tuple[BaseAdapter | ProjectAdapter, float]],
        normalize_weights: bool = True,
    ) -> None:
        """Initialize AdapterStack.

        Args:
            adapters: List of (adapter, weight) tuples
            normalize_weights: If True, normalize weights to sum to 1.0
        """
        if not adapters:
            raise AdapterError("AdapterStack requires at least one adapter")

        # Create stacked adapters
        self._adapters: list[StackedAdapter] = []
        for adapter, weight in adapters:
            self._adapters.append(StackedAdapter(adapter=adapter, weight=weight))

        if normalize_weights:
            self._normalize_weights()

        self._validate()
        self._primary_adapter = self._get_primary_adapter()

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = sum(a.weight for a in self._adapters)
        if total > 0:
            for a in self._adapters:
                a.weight /= total

    def _validate(self) -> None:
        """Validate the adapter stack."""
        total = sum(a.weight for a in self._adapters)
        if abs(total - 1.0) > 0.001:
            raise AdapterError(
                f"Adapter weights must sum to 1.0, got {total}",
                {"weights": [(a.adapter.name, a.weight) for a in self._adapters]},
            )

    def _get_primary_adapter(self) -> StackedAdapter:
        """Get the adapter with the highest weight."""
        return max(self._adapters, key=lambda a: a.weight)

    @property
    def adapters(self) -> list[tuple[BaseAdapter | ProjectAdapter, float]]:
        """Get list of (adapter, weight) tuples."""
        return [(a.adapter, a.weight) for a in self._adapters]

    @property
    def primary(self) -> BaseAdapter | ProjectAdapter:
        """Get the primary (highest-weighted) adapter."""
        return self._primary_adapter.adapter

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> str:
        """Generate text using the adapter stack.

        Currently uses the highest-weighted adapter for generation.
        Future versions may implement true adapter blending.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty

        Returns:
            Generated text
        """
        # Use primary adapter for generation
        # TODO: Implement true adapter stacking when MLX-lm supports it
        logger.debug(
            f"Generating with primary adapter '{self._primary_adapter.adapter.name}' "
            f"(weight: {self._primary_adapter.weight:.0%})"
        )

        return self._primary_adapter.adapter.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

    def add_adapter(
        self,
        adapter: BaseAdapter | ProjectAdapter,
        weight: float,
        normalize: bool = True,
    ) -> None:
        """Add an adapter to the stack.

        Args:
            adapter: Adapter to add
            weight: Weight for the adapter
            normalize: If True, re-normalize all weights after adding
        """
        self._adapters.append(StackedAdapter(adapter=adapter, weight=weight))

        if normalize:
            self._normalize_weights()

        self._validate()
        self._primary_adapter = self._get_primary_adapter()

    def remove_adapter(self, name: str, normalize: bool = True) -> bool:
        """Remove an adapter from the stack by name.

        Args:
            name: Name of the adapter to remove
            normalize: If True, re-normalize weights after removing

        Returns:
            True if adapter was found and removed
        """
        for i, stacked in enumerate(self._adapters):
            if stacked.adapter.name == name:
                self._adapters.pop(i)

                if self._adapters:
                    if normalize:
                        self._normalize_weights()
                    self._validate()
                    self._primary_adapter = self._get_primary_adapter()

                return True

        return False

    def set_weight(self, name: str, weight: float, normalize: bool = True) -> bool:
        """Update the weight of an adapter.

        Args:
            name: Name of the adapter
            weight: New weight
            normalize: If True, re-normalize all weights after updating

        Returns:
            True if adapter was found and updated
        """
        for stacked in self._adapters:
            if stacked.adapter.name == name:
                stacked.weight = weight

                if normalize:
                    self._normalize_weights()
                self._validate()
                self._primary_adapter = self._get_primary_adapter()

                return True

        return False

    def get_config(self) -> AdapterStackConfig:
        """Get the stack configuration."""
        return AdapterStackConfig(
            adapters=[(a.adapter.name, a.weight) for a in self._adapters]
        )

    @classmethod
    def from_config(
        cls,
        config: AdapterStackConfig,
        adapter_lookup: dict[str, BaseAdapter | ProjectAdapter],
    ) -> AdapterStack:
        """Create AdapterStack from configuration.

        Args:
            config: Stack configuration
            adapter_lookup: Dictionary mapping adapter names to instances

        Returns:
            AdapterStack instance
        """
        adapters = []
        for name, weight in config.adapters:
            if name not in adapter_lookup:
                raise AdapterError(
                    f"Adapter not found: {name}",
                    {"available": list(adapter_lookup.keys())},
                )
            adapters.append((adapter_lookup[name], weight))

        return cls(adapters, normalize_weights=False)

    def __len__(self) -> int:
        return len(self._adapters)

    def __repr__(self) -> str:
        adapters_str = ", ".join(
            f"{a.adapter.name}:{a.weight:.0%}" for a in self._adapters
        )
        return f"AdapterStack([{adapters_str}])"
