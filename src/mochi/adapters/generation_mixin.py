"""Generation mixin for shared adapter logic.

Provides common generation functionality used by both BaseAdapter and ProjectAdapter.
This eliminates code duplication and ensures consistent behavior across adapter types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..core.exceptions import AdapterError

if TYPE_CHECKING:
    pass

# Constants for generation parameters
REPETITION_PENALTY_CONTEXT_SIZE = 200  # Increased from 50 to catch longer repetitions
EOS_SUPPRESSION_LOGIT_PENALTY = -1000.0


class GenerationMixin:
    """Mixin class providing common generation logic for adapters.

    This mixin requires the implementing class to have:
    - _model: The loaded MLX model
    - _tokenizer: The tokenizer for the model
    - _loaded: Boolean indicating if model is loaded
    - _ensure_loaded(): Method to ensure model is loaded
    - name: Property returning the adapter name
    """

    _model: Any
    _tokenizer: Any
    _loaded: bool

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded. Must be implemented by subclass."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Get adapter name. Must be implemented by subclass."""
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        min_tokens: int = 0,
        temperature: float = 0.1,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> str:
        """Generate text using the adapter.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            min_tokens: Minimum tokens before allowing EOS (prevents short outputs)
            temperature: Sampling temperature
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty

        Returns:
            Generated text
        """
        self._ensure_loaded()

        try:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_repetition_penalty, make_sampler

            sampler = make_sampler(temp=temperature, top_p=top_p)

            # Build logits processors
            logits_processors = []

            # Add min_tokens processor to prevent early EOS
            if min_tokens > 0:
                logits_processors.append(
                    self._make_min_tokens_processor(min_tokens)
                )

            # Add repetition penalty
            if repetition_penalty != 1.0:
                logits_processors.append(
                    make_repetition_penalty(
                        penalty=repetition_penalty,
                        context_size=REPETITION_PENALTY_CONTEXT_SIZE,
                    )
                )

            result = generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors if logits_processors else None,
            )
            return result

        except Exception as e:
            raise AdapterError(
                "Generation failed",
                {"error": str(e), "adapter": self.name},
            ) from e

    def _make_min_tokens_processor(self, min_tokens: int):
        """Create a logits processor that suppresses EOS until min_tokens.

        This processor prevents the model from generating EOS (end-of-sequence)
        token until at least `min_tokens` have been generated. Useful for
        ensuring substantive outputs for analysis or explanation tasks.

        Note: tokens_generated uses a list to enable mutation within the closure.
        This counter persists across the entire generation session and increments
        with each token generated.

        Args:
            min_tokens: Minimum tokens before allowing EOS

        Returns:
            Logits processor function compatible with mlx_lm.generate()
        """
        import logging
        import mlx.core as mx

        logger = logging.getLogger(__name__)

        # Collect all potential stop tokens to suppress
        stop_token_ids = set()

        # Add the primary EOS token
        if self._tokenizer.eos_token_id is not None:
            stop_token_ids.add(self._tokenizer.eos_token_id)

        # Add other common stop tokens that models might use
        vocab = self._tokenizer.get_vocab()
        for stop_token in ['<|endoftext|>', '</s>', '<|im_end|>', '<|eot_id|>']:
            if stop_token in vocab:
                stop_token_ids.add(vocab[stop_token])

        # Also try to get eos_token_ids from tokenizer (used by mlx_lm)
        if hasattr(self._tokenizer, 'eos_token_ids'):
            for tid in self._tokenizer.eos_token_ids:
                stop_token_ids.add(tid)

        logger.debug(f"[min_tokens_processor] min_tokens={min_tokens}, stop_token_ids={stop_token_ids}")

        tokens_generated = [0]  # Mutable container for closure state

        def processor(tokens: mx.array, logits: mx.array) -> mx.array:
            tokens_generated[0] += 1
            if tokens_generated[0] < min_tokens:
                # Suppress all stop tokens by adding large negative penalty
                # Note: logits is 2D (1, vocab_size), use [..., token_id] for correct indexing
                for token_id in stop_token_ids:
                    logits = logits.at[..., token_id].add(EOS_SUPPRESSION_LOGIT_PENALTY)
            return logits

        return processor
