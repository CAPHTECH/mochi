#!/usr/bin/env python3
"""Training script for different model presets.

Usage:
    # Train with Qwen3-Coder (default)
    python scripts/train_model.py

    # Train with Qwen3-Coder (explicit)
    python scripts/train_model.py --preset qwen3-coder

    # Train with GPT-OSS
    python scripts/train_model.py --preset gpt-oss

    # With offline mode (recommended)
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/train_model.py --preset qwen3-coder
"""

import argparse
import os
from pathlib import Path

from mochi.training.trainer import LoRAConfig, MochiTrainer, MODEL_PRESETS


def main():
    parser = argparse.ArgumentParser(description="Train domain-specific SLM")
    parser.add_argument(
        "--preset",
        type=str,
        default="qwen3-coder",
        choices=list(MODEL_PRESETS.keys()),
        help="Model preset to use",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="./data/train.jsonl",
        help="Path to training data",
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        default=None,
        help="Path to evaluation data (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: ./output/{preset})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=200,
        help="Save checkpoint every N steps",
    )

    args = parser.parse_args()

    # Set output directory
    output_dir = args.output_dir or f"./output/{args.preset}"

    print(f"=== Training with preset: {args.preset} ===")
    print(f"Output directory: {output_dir}")
    print(f"Train file: {args.train_file}")
    print(f"Eval file: {args.eval_file or 'None'}")

    # Create config from preset
    config = LoRAConfig.from_preset(
        args.preset,
        output_dir=output_dir,
        num_epochs=args.epochs,
        save_steps=args.save_steps,
    )

    print(f"\nModel: {config.base_model}")
    print(f"Is MoE: {config.is_moe}")
    print(f"Max seq length: {config.max_seq_length}")

    # Check if offline mode is set
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        print("\nOffline mode enabled - using cached models only")

    # Create trainer and run
    trainer = MochiTrainer(config)
    adapter_path = trainer.train(
        train_file=args.train_file,
        eval_file=args.eval_file,
    )

    print(f"\n=== Training complete ===")
    print(f"Adapter saved to: {adapter_path}")


if __name__ == "__main__":
    main()
