"""Evaluation module for comparing base model vs fine-tuned model."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class EvaluationResult:
    """Result of a single evaluation."""

    prompt: str
    base_output: str
    finetuned_output: str
    expected: str | None = None


class ModelEvaluator:
    """Evaluate and compare base model vs fine-tuned model."""

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-Coder-1.5B",
        finetuned_model_path: str | None = None,
    ) -> None:
        """Initialize evaluator with model paths."""
        self.base_model_name = base_model_name
        self.finetuned_model_path = finetuned_model_path
        self.base_model = None
        self.finetuned_model = None
        self.tokenizer = None

    def setup(self) -> None:
        """Load models for evaluation."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.base_model.eval()

        # Load fine-tuned model if provided
        if self.finetuned_model_path:
            self.finetuned_model = AutoModelForCausalLM.from_pretrained(
                self.finetuned_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.finetuned_model.eval()

    def generate(
        self,
        model: AutoModelForCausalLM,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Generate completion from a model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the generated part
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def evaluate_prompt(
        self,
        prompt: str,
        expected: str | None = None,
        max_new_tokens: int = 256,
    ) -> EvaluationResult:
        """Evaluate a single prompt with both models."""
        if self.base_model is None:
            self.setup()

        base_output = self.generate(self.base_model, prompt, max_new_tokens)

        finetuned_output = ""
        if self.finetuned_model:
            finetuned_output = self.generate(self.finetuned_model, prompt, max_new_tokens)

        return EvaluationResult(
            prompt=prompt,
            base_output=base_output,
            finetuned_output=finetuned_output,
            expected=expected,
        )

    def evaluate_file(
        self,
        eval_file: str | Path,
        output_file: str | Path | None = None,
        max_samples: int = 50,
    ) -> list[EvaluationResult]:
        """
        Evaluate prompts from a JSONL file.

        Args:
            eval_file: Path to evaluation JSONL file (Alpaca format)
            output_file: Path to save results
            max_samples: Maximum number of samples to evaluate
        """
        results: list[EvaluationResult] = []

        with open(eval_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break

                example = json.loads(line)

                # Create prompt (without the expected output)
                instruction = example.get("instruction", "")
                input_text = example.get("input", "")
                expected = example.get("output", "")

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

                result = self.evaluate_prompt(prompt, expected)
                results.append(result)

        if output_file:
            self._save_results(results, output_file)

        return results

    def _save_results(self, results: list[EvaluationResult], output_file: str | Path) -> None:
        """Save evaluation results to JSON."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [
            {
                "prompt": r.prompt,
                "base_output": r.base_output,
                "finetuned_output": r.finetuned_output,
                "expected": r.expected,
            }
            for r in results
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def compare_outputs(results: list[EvaluationResult]) -> dict:
    """Generate comparison statistics."""
    stats = {
        "total": len(results),
        "base_avg_length": 0,
        "finetuned_avg_length": 0,
    }

    if not results:
        return stats

    base_lengths = [len(r.base_output) for r in results]
    finetuned_lengths = [len(r.finetuned_output) for r in results if r.finetuned_output]

    stats["base_avg_length"] = sum(base_lengths) / len(base_lengths)
    if finetuned_lengths:
        stats["finetuned_avg_length"] = sum(finetuned_lengths) / len(finetuned_lengths)

    return stats
