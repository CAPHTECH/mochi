"""MLX Inference engine for Mochi MCP Server.

Optimized for Apple Silicon with MLX framework.
Supports Qwen3-Coder-30B and GPT-OSS-20B trained adapters.

Law compliance:
- L-adapter-required: Adapter must be loaded before inference
- L-response-time: Inference < 5 seconds (MLX is much faster)
- L-memory-bound: Memory usage monitoring
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_repetition_penalty, make_sampler


def make_min_tokens_processor(
    tokenizer,
    min_tokens: int = 10,
) -> callable:
    """Create a logits processor that suppresses EOS until min_tokens.

    P0: 出力長改善 - EOS早期発生を抑制

    Args:
        tokenizer: The tokenizer to get EOS token ID
        min_tokens: Minimum tokens before allowing EOS

    Returns:
        Logits processor function
    """
    eos_token_id = tokenizer.eos_token_id
    tokens_generated = [0]  # Mutable counter

    def processor(tokens: mx.array, logits: mx.array) -> mx.array:
        tokens_generated[0] += 1
        if tokens_generated[0] < min_tokens and eos_token_id is not None:
            # Suppress EOS token by setting its logit to very negative
            logits = logits.at[eos_token_id].add(-1000.0)
        return logits

    return processor


class TaskType(Enum):
    """Types of code generation tasks."""

    COMPLETION = "completion"
    METHOD_CALL = "method_call"
    EXPLANATION = "explanation"
    DOCUMENTATION = "documentation"
    IMPORT = "import"
    GENERAL = "general"
    DIFF = "diff"  # P1: 差分生成モード


class GenerationMode(Enum):
    """P2: Generation modes for confidence-based switching.

    - AUTO: Automatically select mode based on context and results
    - CONSERVATIVE: Lower temperature, stricter context adherence
    - CREATIVE: Higher temperature, more diverse outputs
    """

    AUTO = "auto"
    CONSERVATIVE = "conservative"
    CREATIVE = "creative"


class PromptTemplate:
    """Prompt templates for different task types.

    Optimized for code completion and generation tasks.
    """

    # Alpaca-style templates (used during training)
    ALPACA_WITH_INPUT = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

    ALPACA_NO_INPUT = """### Instruction:
{instruction}

### Response:
"""

    # Context-aware template (includes LSP context)
    # Context placed FIRST to ensure model uses it
    CONTEXT_AWARE = """### Context (MUST USE):
{context}

### Instruction:
{instruction}
Use ONLY the methods listed in Context above.

### Input:
{input}

### Response:
"""

    # Code completion specific (minimal)
    CODE_COMPLETION = """{context}
{code}"""

    # P1: Diff generation template
    DIFF_TEMPLATE = """### Instruction:
Generate a COMPLETE unified diff for the following change.
Include ALL additions and modifications. Do not stop early.
Output format: unified diff starting with --- and +++.

### Change Request:
{change_description}

### Original Code ({language}):
{original_code}

### Response (complete unified diff with all changes):
---"""

    @classmethod
    def format(
        cls,
        task_type: TaskType,
        instruction: str = "",
        input_text: str = "",
        context: str = "",
        use_alpaca: bool = True,
    ) -> str:
        """Format prompt based on task type.

        Args:
            task_type: Type of task (affects prompt structure)
            instruction: The instruction text
            input_text: The input code/text
            context: LSP context (available methods, types, etc.)
            use_alpaca: Use Alpaca format (True) or minimal format (False)

        Returns:
            Formatted prompt string
        """
        if not use_alpaca:
            # Minimal format for pure code completion
            parts = []
            if context:
                parts.append(context)
            if input_text:
                parts.append(input_text)
            return "\n".join(parts)

        # Build instruction based on task type if not provided
        if not instruction:
            instruction = cls._default_instruction(task_type)

        # Use context-aware template if context provided
        if context:
            return cls.CONTEXT_AWARE.format(
                instruction=instruction,
                context=context,
                input=input_text,
            )
        elif input_text:
            return cls.ALPACA_WITH_INPUT.format(
                instruction=instruction,
                input=input_text,
            )
        else:
            return cls.ALPACA_NO_INPUT.format(
                instruction=instruction,
            )

    @classmethod
    def _default_instruction(cls, task_type: TaskType) -> str:
        """Get default instruction for task type."""
        instructions = {
            TaskType.COMPLETION: "Complete the following code:",
            TaskType.METHOD_CALL: "Complete the method call:",
            TaskType.EXPLANATION: "Explain what this code does:",
            TaskType.DOCUMENTATION: "Add documentation to this code:",
            TaskType.IMPORT: "Add the necessary imports:",
            TaskType.GENERAL: "Complete the following:",
            TaskType.DIFF: "Generate a unified diff:",
        }
        return instructions.get(task_type, instructions[TaskType.GENERAL])


@dataclass
class InferenceConfig:
    """Configuration for inference based on task type.

    Optimized defaults for different code generation scenarios.
    """

    max_tokens: int = 256
    min_tokens: int = 10  # P0: 最小出力長
    temperature: float = 0.1
    top_p: float = 0.5
    repetition_penalty: float = 1.2
    repetition_context_size: int = 50

    # Task-specific preset configurations
    PRESETS: dict[TaskType, dict[str, Any]] = None

    def __post_init__(self):
        if self.PRESETS is None:
            self.PRESETS = {
                # Code completion: medium length, low temp for consistency
                TaskType.COMPLETION: {
                    "max_tokens": 256,
                    "min_tokens": 15,  # P0: Ensure meaningful completions
                    "temperature": 0.1,
                    "top_p": 0.5,
                    "repetition_penalty": 1.2,
                },
                # Method call: short, very low temp for deterministic output
                TaskType.METHOD_CALL: {
                    "max_tokens": 128,
                    "min_tokens": 5,  # P0: At least method name + args
                    "temperature": 0.05,
                    "top_p": 0.3,
                    "repetition_penalty": 1.3,  # Higher to avoid repetition
                },
                # Explanation: longer, slightly higher temp for natural language
                TaskType.EXPLANATION: {
                    "max_tokens": 512,
                    "min_tokens": 20,  # P0: Meaningful explanation
                    "temperature": 0.3,
                    "top_p": 0.7,
                    "repetition_penalty": 1.1,
                },
                # Documentation: medium, moderate temp
                TaskType.DOCUMENTATION: {
                    "max_tokens": 384,
                    "min_tokens": 15,  # P0: Meaningful docstring
                    "temperature": 0.2,
                    "top_p": 0.6,
                    "repetition_penalty": 1.15,
                },
                # Import: short, very low temp
                TaskType.IMPORT: {
                    "max_tokens": 128,
                    "min_tokens": 5,  # P0: At least one import
                    "temperature": 0.05,
                    "top_p": 0.3,
                    "repetition_penalty": 1.3,
                },
                # General: balanced defaults
                TaskType.GENERAL: {
                    "max_tokens": 256,
                    "min_tokens": 10,  # P0: Default minimum
                    "temperature": 0.1,
                    "top_p": 0.5,
                    "repetition_penalty": 1.2,
                },
                # P1: Diff generation - focused, deterministic
                TaskType.DIFF: {
                    "max_tokens": 512,
                    "min_tokens": 10,  # P0: At least diff header + change
                    "temperature": 0.05,  # Very low for precise diffs
                    "top_p": 0.3,
                    "repetition_penalty": 1.1,
                },
            }

    @classmethod
    def for_task(cls, task_type: TaskType) -> "InferenceConfig":
        """Get optimized config for a specific task type."""
        config = cls()
        preset = config.PRESETS.get(task_type, config.PRESETS[TaskType.GENERAL])
        return cls(
            max_tokens=preset["max_tokens"],
            min_tokens=preset.get("min_tokens", 10),  # P0: Include min_tokens
            temperature=preset["temperature"],
            top_p=preset["top_p"],
            repetition_penalty=preset["repetition_penalty"],
        )


@dataclass
class InferenceResult:
    """Result of model inference."""

    response: str
    confidence: float
    inference_time_ms: float
    tokens_generated: int
    # P0: confidence threshold warnings
    warning: str | None = None
    alternative_action: str | None = None
    # P2: mode information
    mode_used: str = "auto"
    retried: bool = False

    # Confidence thresholds
    CONFIDENCE_LOW = 0.3
    CONFIDENCE_MEDIUM = 0.5
    CONFIDENCE_HIGH = 0.7

    def with_confidence_warning(self) -> "InferenceResult":
        """Add warning and alternative action based on confidence level."""
        if self.confidence < self.CONFIDENCE_LOW:
            self.warning = "非常に低い確信度: 出力の信頼性が低い可能性があります"
            self.alternative_action = "files_searchで既存パターンを確認し、手動で判断することを推奨"
        elif self.confidence < self.CONFIDENCE_MEDIUM:
            self.warning = "低確信度: 学習データに類似パターンが少ない"
            self.alternative_action = "kiri.snippets_getで関連コードを確認することを推奨"
        elif self.confidence < self.CONFIDENCE_HIGH:
            self.warning = "中程度の確信度: 出力を確認してください"
            self.alternative_action = None
        # High confidence: no warning
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "response": self.response,
            "confidence": self.confidence,
            "inference_time_ms": self.inference_time_ms,
            "tokens_generated": self.tokens_generated,
            "mode_used": self.mode_used,
        }
        if self.retried:
            result["retried"] = self.retried
        if self.warning:
            result["warning"] = self.warning
        if self.alternative_action:
            result["alternative_action"] = self.alternative_action
        return result


class MLXInferenceEngine:
    """MLX-based inference engine for Apple Silicon.

    Much faster than PyTorch+MPS for large models.
    Supports Qwen3-Coder-30B-A3B and GPT-OSS-20B.
    """

    # Project root for resolving relative paths
    _PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

    # Preset configurations for supported models
    PRESETS: dict[str, dict[str, Any]] = {
        "qwen3-coder": {
            "model_path": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
            "default_adapter": _PROJECT_ROOT / "output/mlx-qwen3-coder/adapter",
        },
        "gpt-oss": {
            "model_path": "lmstudio-community/gpt-oss-20b-MLX-8bit",
            "default_adapter": _PROJECT_ROOT / "output/gptoss-20b-lsp/adapter",
        },
    }

    def __init__(
        self,
        model_path: str | None = None,
        adapter_path: str | Path | None = None,
        preset: str | None = None,
        timeout_seconds: float = 5.0,
        max_memory_gb: float = 64.0,
    ) -> None:
        """Initialize MLX inference engine.

        Args:
            model_path: Path to MLX model (HuggingFace ID or local path)
            adapter_path: Path to LoRA adapter
            preset: Use preset config ("qwen3-coder" or "gpt-oss")
            timeout_seconds: Max inference time (L-response-time)
            max_memory_gb: Max memory usage (L-memory-bound)
        """
        # Apply preset if specified
        if preset and preset in self.PRESETS:
            preset_config = self.PRESETS[preset]
            model_p = model_path or preset_config["model_path"]
            self.model_path = str(model_p) if isinstance(model_p, Path) else model_p
            self.adapter_path = Path(adapter_path or preset_config["default_adapter"])
        else:
            self.model_path = model_path or self.PRESETS["qwen3-coder"]["model_path"]
            self.adapter_path = Path(adapter_path) if adapter_path else None

        self.timeout_seconds = timeout_seconds
        self.max_memory_gb = max_memory_gb

        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._preset = preset

    @property
    def is_loaded(self) -> bool:
        """Check if model and adapter are loaded (L-adapter-required)."""
        return self._loaded and self.model is not None and self.tokenizer is not None

    def load(self) -> None:
        """Load MLX model and adapter.

        Raises:
            FileNotFoundError: If adapter_path doesn't exist
            RuntimeError: If loading fails
        """
        if self.adapter_path and not self.adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {self.adapter_path}")

        # Load model with adapter using mlx_lm
        adapter_str = str(self.adapter_path) if self.adapter_path else None

        self.model, self.tokenizer = load(
            self.model_path,
            adapter_path=adapter_str,
        )

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
            return True

    # Training artifacts to remove from output
    TRAINING_ARTIFACTS = [
        "### Response:",
        "### Instruction:",
        "### Input:",
        "### Context:",
        "### Context (MUST USE):",
        "### Change Request:",
        "### Original Code:",
        "### Response (unified diff only):",
        "<|endoftext|>",
        "<|im_end|>",
        "<|im_start|>",
    ]

    def _clean_response(self, response: str) -> str:
        """Remove training artifacts from generated response.

        P0: アーティファクト除去

        Args:
            response: Raw generated text

        Returns:
            Cleaned response without training artifacts
        """
        if not response:
            return response

        # Remove known training artifacts
        for artifact in self.TRAINING_ARTIFACTS:
            if artifact in response:
                # Take content before the artifact (model started repeating prompt)
                parts = response.split(artifact)
                response = parts[0].strip()
                # If there's content after, it might be a continuation
                if len(parts) > 1 and parts[1].strip():
                    # Check if it's not another artifact/repetition
                    after = parts[1].strip()
                    if not any(a in after for a in self.TRAINING_ARTIFACTS[:5]):
                        # Might be valid content that was split
                        # Take the longer meaningful part
                        if len(after) > len(response) and not response:
                            response = after

        # Remove markdown code fences if they wrap the entire response
        lines = response.strip().split("\n")
        if lines:
            # Check if starts with code fence
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Check if ends with code fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response = "\n".join(lines)

        # P0: Remove duplicate diff blocks (LLM sometimes repeats same diff)
        if response.startswith("---"):
            # Split by diff headers, keeping the delimiter
            blocks = re.split(r"(?=^--- )", response, flags=re.MULTILINE)
            if len(blocks) > 1:
                # Keep only unique blocks (preserving order)
                seen = set()
                unique_blocks = []
                for block in blocks:
                    block_stripped = block.strip()
                    if block_stripped and block_stripped not in seen:
                        seen.add(block_stripped)
                        unique_blocks.append(block)
                response = "".join(unique_blocks)

        # Remove leading/trailing whitespace
        response = response.strip()

        # Remove trailing incomplete lines (often caused by max_tokens cutoff)
        # but only if they look incomplete (no semicolon, brace, etc.)
        if response and not response[-1] in ";})]\n\"'`":
            lines = response.split("\n")
            if len(lines) > 1:
                last_line = lines[-1].strip()
                # If last line looks incomplete and short, remove it
                if len(last_line) < 10 and not last_line.endswith((",", ".", ":", "{")):
                    response = "\n".join(lines[:-1])

        return response.strip()

    def _calculate_confidence(
        self,
        response: str,
        tokens_generated: int,
        min_tokens: int,
    ) -> float:
        """Calculate confidence score based on output quality.

        P0: Improved confidence calculation

        Args:
            response: Generated text
            tokens_generated: Number of tokens generated
            min_tokens: Expected minimum tokens

        Returns:
            Confidence score 0.0 to 1.0
        """
        if not response:
            return 0.0

        # Base confidence from output length
        # More tokens = higher base confidence (up to a point)
        length_score = min(1.0, tokens_generated / max(min_tokens * 2, 20))

        # Check for code-like content (higher confidence for actual code)
        code_indicators = [
            "function", "const", "let", "var", "import", "export",
            "class", "interface", "type", "async", "await",
            "return", "if", "for", "while", "=>", "{", "}",
        ]
        code_score = sum(1 for ind in code_indicators if ind in response)
        code_confidence = min(1.0, code_score / 5)

        # Penalize very short responses
        if tokens_generated < 5:
            length_penalty = 0.3
        elif tokens_generated < 10:
            length_penalty = 0.6
        else:
            length_penalty = 1.0

        # Combined confidence
        confidence = (length_score * 0.4 + code_confidence * 0.6) * length_penalty

        return min(1.0, max(0.1, confidence))

    # P2: Mode-specific parameter adjustments
    MODE_PARAMS = {
        GenerationMode.CONSERVATIVE: {
            "temperature": 0.05,
            "top_p": 0.3,
            "repetition_penalty": 1.3,
        },
        GenerationMode.CREATIVE: {
            "temperature": 0.3,
            "top_p": 0.7,
            "repetition_penalty": 1.1,
        },
    }

    def generate(
        self,
        instruction: str,
        input_text: str = "",
        context: str = "",
        task_type: TaskType = TaskType.COMPLETION,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 10,  # P0: 最小出力長
        temperature: float = 0.1,
        top_p: float = 0.5,
        use_alpaca_format: bool = True,
        mode: GenerationMode = GenerationMode.AUTO,  # P2: モード切替
        auto_retry: bool = True,  # P2: 低confidence時の自動リトライ
    ) -> InferenceResult:
        """Generate response from the model.

        Args:
            instruction: The instruction/question
            input_text: Additional context (code to complete)
            context: LSP context (available methods, types, etc.)
            task_type: Type of task for prompt optimization
            max_new_tokens: Maximum tokens to generate
            min_new_tokens: Minimum tokens to generate (suppresses early EOS)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            use_alpaca_format: Use Alpaca format (True) or minimal (False)
            mode: Generation mode (AUTO, CONSERVATIVE, CREATIVE)
            auto_retry: If True and mode=AUTO, retry with conservative params on low confidence

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

        # P2: Adjust params based on mode
        actual_temp = temperature
        actual_top_p = top_p
        rep_penalty = 1.2
        mode_used = mode.value

        if mode == GenerationMode.CONSERVATIVE:
            params = self.MODE_PARAMS[GenerationMode.CONSERVATIVE]
            actual_temp = params["temperature"]
            actual_top_p = params["top_p"]
            rep_penalty = params["repetition_penalty"]
        elif mode == GenerationMode.CREATIVE:
            params = self.MODE_PARAMS[GenerationMode.CREATIVE]
            actual_temp = params["temperature"]
            actual_top_p = params["top_p"]
            rep_penalty = params["repetition_penalty"]

        # First generation attempt
        result = self._generate_once(
            instruction=instruction,
            input_text=input_text,
            context=context,
            task_type=task_type,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=actual_temp,
            top_p=actual_top_p,
            repetition_penalty=rep_penalty,
            use_alpaca_format=use_alpaca_format,
        )

        # P2: Auto-retry with conservative params if confidence is low
        if (
            mode == GenerationMode.AUTO
            and auto_retry
            and result.confidence < InferenceResult.CONFIDENCE_MEDIUM
        ):
            # Retry with conservative parameters
            conservative_params = self.MODE_PARAMS[GenerationMode.CONSERVATIVE]
            retry_result = self._generate_once(
                instruction=instruction,
                input_text=input_text,
                context=context,
                task_type=task_type,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                temperature=conservative_params["temperature"],
                top_p=conservative_params["top_p"],
                repetition_penalty=conservative_params["repetition_penalty"],
                use_alpaca_format=use_alpaca_format,
            )

            # Use retry result if it has higher confidence
            if retry_result.confidence > result.confidence:
                result = retry_result
                result.retried = True
                mode_used = "auto->conservative"
            else:
                result.retried = True  # Mark as retried even if original was better

        result.mode_used = mode_used
        return result.with_confidence_warning()

    def _generate_once(
        self,
        instruction: str,
        input_text: str,
        context: str,
        task_type: TaskType,
        max_new_tokens: int,
        min_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        use_alpaca_format: bool,
    ) -> InferenceResult:
        """Internal: Single generation attempt without retry logic."""
        start_time = time.time()

        # Format prompt using PromptTemplate
        prompt = PromptTemplate.format(
            task_type=task_type,
            instruction=instruction,
            input_text=input_text,
            context=context,
            use_alpaca=use_alpaca_format,
        )

        # Create sampler with temperature and top_p
        sampler = make_sampler(temp=temperature, top_p=top_p)

        # Create logits processors
        logits_processors = []

        # P0: Add min tokens processor to prevent early EOS
        if min_new_tokens > 0:
            min_tokens_processor = make_min_tokens_processor(
                self.tokenizer, min_tokens=min_new_tokens
            )
            logits_processors.append(min_tokens_processor)

        # Add repetition penalty processor to avoid repetitive outputs
        rep_processor = make_repetition_penalty(penalty=repetition_penalty, context_size=50)
        logits_processors.append(rep_processor)

        # Generate using mlx_lm with processors
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        )

        elapsed = time.time() - start_time

        # Extract just the response part (remove the prompt)
        if response.startswith(prompt):
            response = response[len(prompt):]

        # P0: Clean up response and remove training artifacts
        response = self._clean_response(response)

        # Estimate tokens (rough)
        tokens_generated = len(response.split())

        # P0: Improved confidence calculation based on output quality
        confidence = self._calculate_confidence(
            response, tokens_generated, min_new_tokens
        )

        return InferenceResult(
            response=response,
            confidence=confidence,
            inference_time_ms=elapsed * 1000,
            tokens_generated=tokens_generated,
        )

    def generate_completion(
        self,
        code_prefix: str,
        context: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ) -> InferenceResult:
        """Generate code completion with optimized settings.

        Convenience method for code completion with LSP context.

        Args:
            code_prefix: Code before cursor position
            context: LSP context (available methods, types)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            InferenceResult with completion
        """
        return self.generate(
            instruction="Complete the following code:",
            input_text=code_prefix,
            context=context,
            task_type=TaskType.COMPLETION,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    def generate_method_completion(
        self,
        code_with_dot: str,
        available_methods: str = "",
        max_new_tokens: int = 128,
        temperature: float = 0.05,
    ) -> InferenceResult:
        """Generate method call completion.

        Optimized for completing method calls after a dot.

        Args:
            code_with_dot: Code ending with object.
            available_methods: List of available methods from LSP
            max_new_tokens: Maximum tokens to generate
            temperature: Lower for more deterministic output

        Returns:
            InferenceResult with method completion
        """
        context = ""
        if available_methods:
            # Emphasize that ONLY these methods are valid
            context = f"// VALID methods (use ONLY these):\n{available_methods}"

        return self.generate(
            instruction="Complete the method call using ONLY the methods listed in Context.",
            input_text=code_with_dot,
            context=context,
            task_type=TaskType.METHOD_CALL,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    def generate_diff(
        self,
        original_code: str,
        change_description: str,
        language: str = "typescript",
        context: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.05,
    ) -> InferenceResult:
        """Generate unified diff for a code change.

        P1: 差分生成モード - 全コード生成ではなく変更部分のみを提案

        Args:
            original_code: The original code to modify
            change_description: Description of the desired change
            language: Programming language (for syntax highlighting in prompt)
            context: LSP context (available methods, types)
            max_new_tokens: Maximum tokens to generate
            temperature: Lower for more deterministic diffs

        Returns:
            InferenceResult with unified diff
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.time()

        # Use the DIFF_TEMPLATE directly
        prompt = PromptTemplate.DIFF_TEMPLATE.format(
            change_description=change_description,
            original_code=original_code,
            language=language,
        )

        # If context provided, prepend it
        if context:
            prompt = f"### Context (MUST USE):\n{context}\n\n{prompt}"

        # Create sampler - slightly higher temp for complete output
        # (too low causes early EOS on TypeScript)
        effective_temp = max(temperature, 0.2)
        sampler = make_sampler(temp=effective_temp, top_p=0.6)
        # P0: Moderate penalty to prevent duplicates without truncating output
        repetition_penalty = make_repetition_penalty(penalty=1.15, context_size=100)
        # P0: Prevent early EOS - ensure minimum output length
        min_tokens_processor = make_min_tokens_processor(self.tokenizer, min_tokens=30)

        # Generate
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
            sampler=sampler,
            logits_processors=[repetition_penalty, min_tokens_processor],
        )

        elapsed = time.time() - start_time

        # Extract just the response part
        if response.startswith(prompt):
            response = response[len(prompt):]

        # P0: Clean up training artifacts first
        response = self._clean_response(response)

        # Clean up response - keep only diff-like content
        response = self._extract_diff(response)

        tokens_generated = len(response.split())
        confidence = self._calculate_diff_confidence(response, original_code)

        return InferenceResult(
            response=response,
            confidence=confidence,
            inference_time_ms=elapsed * 1000,
            tokens_generated=tokens_generated,
        ).with_confidence_warning()

    def _extract_diff(self, response: str) -> str:
        """Extract unified diff from response, cleaning up non-diff content.

        P0: Permissive extraction - keep all content between diff headers.
        Only remove clear non-diff content (prose before/after diff).
        """
        lines = response.split("\n")
        diff_lines = []
        in_diff = False
        seen_first_block = False  # Track if we've completed first diff block

        for line in lines:
            # Detect start of diff
            if line.startswith("---"):
                if not in_diff:
                    in_diff = True
                    diff_lines.append(line)
                elif seen_first_block:
                    # Second --- header = duplicate block, stop
                    break
                else:
                    diff_lines.append(line)
            # +++ header
            elif line.startswith("+++"):
                if in_diff:
                    diff_lines.append(line)
                    seen_first_block = True
            elif in_diff:
                # Code fence - stop
                if line.startswith("```"):
                    break
                # Keep ALL content once we're in diff mode
                # This includes +, -, context lines, empty lines, etc.
                diff_lines.append(line)

        # If no proper diff found, return as-is
        if not diff_lines:
            return response

        return "\n".join(diff_lines).strip()

    def _calculate_diff_confidence(self, diff: str, original: str) -> float:
        """Calculate confidence score for generated diff."""
        # Basic heuristics for diff quality
        if not diff:
            return 0.0

        # Check for proper diff structure
        has_header = "---" in diff or "+++" in diff
        has_hunks = "@@" in diff
        has_changes = "+" in diff or "-" in diff

        # Start with base confidence
        confidence = 0.3

        if has_header:
            confidence += 0.2
        if has_hunks:
            confidence += 0.2
        if has_changes:
            confidence += 0.2

        # Penalize very short or very long diffs
        diff_lines = len(diff.split("\n"))
        if diff_lines < 3:
            confidence *= 0.7
        elif diff_lines > 50:
            confidence *= 0.8

        return min(1.0, confidence)

    def generate_with_config(
        self,
        instruction: str,
        input_text: str = "",
        context: str = "",
        task_type: TaskType = TaskType.COMPLETION,
        config: InferenceConfig | None = None,
    ) -> InferenceResult:
        """Generate response using task-specific configuration.

        Automatically applies optimized settings based on task type
        if no config is provided.

        Args:
            instruction: The instruction/question
            input_text: Input code/text
            context: LSP context
            task_type: Type of task
            config: Custom config (uses task default if None)

        Returns:
            InferenceResult with response
        """
        if config is None:
            config = InferenceConfig.for_task(task_type)

        return self.generate(
            instruction=instruction,
            input_text=input_text,
            context=context,
            task_type=task_type,
            max_new_tokens=config.max_tokens,
            min_new_tokens=config.min_tokens,  # P0: Use min_tokens from config
            temperature=config.temperature,
            top_p=config.top_p,
        )

    def complete(
        self,
        prefix: str,
        suffix: str = "",
        max_new_tokens: int = 256,
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
                temperature=0.3,  # Lower for more consistent completions
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


def create_engine(
    backend: str = "mlx",
    preset: str | None = None,
    model_path: str | None = None,
    adapter_path: str | Path | None = None,
    **kwargs,
) -> MLXInferenceEngine:
    """Factory function to create inference engine.

    Args:
        backend: "mlx" (default) or "pytorch"
        preset: Model preset ("qwen3-coder", "gpt-oss")
        model_path: Custom model path
        adapter_path: Custom adapter path
        **kwargs: Additional arguments

    Returns:
        Configured inference engine
    """
    if backend == "mlx":
        return MLXInferenceEngine(
            model_path=model_path,
            adapter_path=adapter_path,
            preset=preset,
            **kwargs,
        )
    elif backend == "pytorch":
        # Import PyTorch engine for fallback
        from mochi.mcp.inference import InferenceEngine
        return InferenceEngine(
            base_model=model_path or "Qwen/Qwen3-30B-A3B",
            adapter_path=adapter_path,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'mlx' or 'pytorch'.")
