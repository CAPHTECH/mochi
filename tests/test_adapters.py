"""Tests for adapter modules.

Tests for GenerationMixin, BaseAdapter, ProjectAdapter, and AdapterStack.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestGenerationMixin:
    """Tests for GenerationMixin class."""

    def test_constants_defined(self):
        """Test that generation constants are defined."""
        from mochi.adapters.generation_mixin import (
            REPETITION_PENALTY_CONTEXT_SIZE,
            EOS_SUPPRESSION_LOGIT_PENALTY,
        )

        assert REPETITION_PENALTY_CONTEXT_SIZE == 200  # Increased to catch longer repetitions
        assert EOS_SUPPRESSION_LOGIT_PENALTY == -1000.0

    def test_generate_signature(self):
        """Test that generate method has correct signature."""
        import inspect
        from mochi.adapters.generation_mixin import GenerationMixin

        sig = inspect.signature(GenerationMixin.generate)
        params = list(sig.parameters.keys())

        assert "prompt" in params
        assert "max_tokens" in params
        assert "min_tokens" in params
        assert "temperature" in params
        assert "top_p" in params
        assert "repetition_penalty" in params

    def test_min_tokens_processor_signature(self):
        """Test that _make_min_tokens_processor method exists."""
        from mochi.adapters.generation_mixin import GenerationMixin

        assert hasattr(GenerationMixin, "_make_min_tokens_processor")


class TestBaseAdapter:
    """Tests for BaseAdapter class."""

    def test_inherits_generation_mixin(self):
        """Test that BaseAdapter inherits from GenerationMixin."""
        from mochi.adapters.base_adapter import BaseAdapter
        from mochi.adapters.generation_mixin import GenerationMixin

        assert issubclass(BaseAdapter, GenerationMixin)

    def test_generate_method_inherited(self):
        """Test that generate method is inherited from mixin."""
        from mochi.adapters.base_adapter import BaseAdapter
        from mochi.adapters.generation_mixin import GenerationMixin

        # The generate method should be the same as the mixin's
        assert BaseAdapter.generate is GenerationMixin.generate


class TestProjectAdapter:
    """Tests for ProjectAdapter class."""

    def test_inherits_generation_mixin(self):
        """Test that ProjectAdapter inherits from GenerationMixin."""
        from mochi.adapters.project_adapter import ProjectAdapter
        from mochi.adapters.generation_mixin import GenerationMixin

        assert issubclass(ProjectAdapter, GenerationMixin)

    def test_generate_method_inherited(self):
        """Test that generate method is inherited from mixin."""
        from mochi.adapters.project_adapter import ProjectAdapter
        from mochi.adapters.generation_mixin import GenerationMixin

        # The generate method should be the same as the mixin's
        assert ProjectAdapter.generate is GenerationMixin.generate


class TestAdapterStack:
    """Tests for AdapterStack class."""

    def test_generate_has_min_tokens(self):
        """Test that AdapterStack.generate accepts min_tokens."""
        import inspect
        from mochi.adapters.adapter_stack import AdapterStack

        sig = inspect.signature(AdapterStack.generate)
        params = list(sig.parameters.keys())

        assert "min_tokens" in params

    def test_generate_default_min_tokens(self):
        """Test that min_tokens defaults to 0."""
        import inspect
        from mochi.adapters.adapter_stack import AdapterStack

        sig = inspect.signature(AdapterStack.generate)
        min_tokens_param = sig.parameters["min_tokens"]

        assert min_tokens_param.default == 0


class TestMinTokensProcessor:
    """Tests for min_tokens logits processor."""

    def test_processor_creation(self):
        """Test that min_tokens processor can be created."""
        from mochi.adapters.generation_mixin import GenerationMixin

        # Create a mock adapter with required attributes
        mock_adapter = MagicMock(spec=GenerationMixin)
        mock_adapter._tokenizer = MagicMock()
        mock_adapter._tokenizer.eos_token_id = 151645  # Common EOS token

        # Bind the method to the mock
        processor_method = GenerationMixin._make_min_tokens_processor
        processor = processor_method(mock_adapter, min_tokens=100)

        assert callable(processor)

    @pytest.mark.skipif(
        True,  # Skip unless MLX is available
        reason="Requires MLX for array operations"
    )
    def test_processor_suppresses_eos(self):
        """Test that processor suppresses EOS before min_tokens."""
        # This test requires MLX to be installed
        pass


class TestAsyncUtils:
    """Tests for async utility functions."""

    def test_run_sync_with_coroutine(self):
        """Test run_sync executes coroutine and returns result."""
        from mochi.core.async_utils import run_sync

        async def async_func():
            return 42

        result = run_sync(async_func())
        assert result == 42

    def test_run_sync_with_exception(self):
        """Test run_sync propagates exceptions."""
        from mochi.core.async_utils import run_sync

        async def async_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            run_sync(async_func())

    def test_run_sync_with_await(self):
        """Test run_sync handles await correctly."""
        import asyncio
        from mochi.core.async_utils import run_sync

        async def async_func():
            await asyncio.sleep(0.001)
            return "done"

        result = run_sync(async_func())
        assert result == "done"
