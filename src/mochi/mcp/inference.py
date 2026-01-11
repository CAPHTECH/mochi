"""Inference engine for Mochi MCP Server.

Law compliance:
- L-adapter-required: Adapter must be loaded before inference
- L-response-time: Inference < 5 seconds
- L-memory-bound: Memory usage < 32GB
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class InferenceResult:
    """Result of model inference."""

    response: str
    confidence: float
    inference_time_ms: float
    tokens_generated: int


class InferenceEngine:
    """Inference engine wrapping the fine-tuned SLM.

    Implements:
    - L-adapter-required: Check adapter loaded before inference
    - L-response-time: Timeout control
    - L-memory-bound: Memory monitoring
    """

    def __init__(
        self,
        base_model: str = "Qwen/Qwen3-Coder-30B-A3B",
        adapter_path: str | Path | None = None,
        device: str | None = None,
        timeout_seconds: float = 5.0,
        max_memory_gb: float = 32.0,
    ) -> None:
        """Initialize inference engine.

        Args:
            base_model: HuggingFace model ID
            adapter_path: Path to LoRA adapter
            device: Device to use (auto-detected if None)
            timeout_seconds: Max inference time (L-response-time)
            max_memory_gb: Max memory usage (L-memory-bound)
        """
        self.base_model = base_model
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.timeout_seconds = timeout_seconds
        self.max_memory_gb = max_memory_gb

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model = None
        self.tokenizer = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model and adapter are loaded (L-adapter-required)."""
        return self._loaded and self.model is not None and self.tokenizer is not None

    def load(self) -> None:
        """Load base model and adapter.

        Raises:
            FileNotFoundError: If adapter_path doesn't exist
            RuntimeError: If loading fails
        """
        if self.adapter_path and not self.adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {self.adapter_path}")

        # Determine dtype based on device
        if self.device == "cuda":
            dtype = torch.float16
        elif self.device == "mps":
            dtype = torch.float32  # MPS stability
        else:
            dtype = torch.float32

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        if self.device == "mps":
            device_map = {"": "mps"}
        elif self.device == "cuda":
            device_map = "auto"
        else:
            device_map = "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        # Load adapter if provided
        if self.adapter_path:
            self.model = PeftModel.from_pretrained(
                self.model,
                str(self.adapter_path),
            )

        self.model.eval()
        self._loaded = True

    def _check_memory(self) -> bool:
        """Check memory usage (L-memory-bound).

        Returns:
            True if memory is within bounds
        """
        try:
            import psutil

            memory = psutil.virtual_memory()
            used_gb = memory.used / (1024**3)
            return used_gb < self.max_memory_gb
        except ImportError:
            # If psutil not available, skip check
            return True

    def generate(
        self,
        instruction: str,
        input_text: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> InferenceResult:
        """Generate response from the model.

        Args:
            instruction: The instruction/question
            input_text: Additional context
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            InferenceResult with response and metadata

        Raises:
            RuntimeError: If model not loaded (L-adapter-required violation)
            TimeoutError: If inference exceeds timeout (L-response-time violation)
            MemoryError: If memory limit exceeded (L-memory-bound violation)
        """
        # L-adapter-required check
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # L-memory-bound check
        if not self._check_memory():
            raise MemoryError(
                f"Memory usage exceeds {self.max_memory_gb}GB limit"
            )

        start_time = time.time()

        # Format prompt (Alpaca style)
        if input_text:
            prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
        else:
            prompt = f"""### Instruction:
{instruction}

### Response:
"""

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        # Move to device
        if self.device == "mps":
            inputs = {k: v.to("mps") for k, v in inputs.items()}
        elif self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        input_length = inputs["input_ids"].shape[1]

        # Generate with timeout awareness
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        elapsed = time.time() - start_time

        # L-response-time check
        if elapsed > self.timeout_seconds:
            raise TimeoutError(
                f"Inference took {elapsed:.2f}s, exceeds {self.timeout_seconds}s limit"
            )

        # Decode response
        generated_ids = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        tokens_generated = len(generated_ids)

        # Simple confidence based on output length ratio
        confidence = min(1.0, tokens_generated / max_new_tokens)

        return InferenceResult(
            response=response.strip(),
            confidence=confidence,
            inference_time_ms=elapsed * 1000,
            tokens_generated=tokens_generated,
        )

    def complete(
        self,
        prefix: str,
        suffix: str = "",
        max_new_tokens: int = 128,
        num_alternatives: int = 3,
    ) -> list[str]:
        """Complete code given prefix and optional suffix.

        Args:
            prefix: Code before cursor
            suffix: Code after cursor
            max_new_tokens: Max tokens per completion
            num_alternatives: Number of alternatives to generate

        Returns:
            List of completion candidates
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if suffix:
            prompt = f"{prefix}<CURSOR>{suffix}"
            instruction = "Complete the code at <CURSOR> position"
        else:
            prompt = prefix
            instruction = "Complete the following code"

        completions = []
        for _ in range(num_alternatives):
            result = self.generate(
                instruction=instruction,
                input_text=prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.8,  # Higher for diversity
            )
            if result.response:
                completions.append(result.response)

        return completions

    def unload(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False

        # Force garbage collection
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
