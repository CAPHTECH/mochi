#!/bin/bash
# Train Base Adapter with Test Patterns
# This script trains the base adapter on combined data (common + test patterns)

set -e

# Configuration
MODEL="mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit"  # Small model for testing
DATA_DIR="data/combined-with-tests"
OUTPUT_DIR="output/base-adapter-with-tests"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=== Training Base Adapter with Test Patterns ==="
echo "Model: $MODEL"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Run MLX-LM LoRA training
/opt/homebrew/bin/python3.11 -m mlx_lm lora \
    --model "$MODEL" \
    --train \
    --data "$DATA_DIR" \
    --adapter-path "$OUTPUT_DIR/adapter" \
    --iters 100 \
    --batch-size 2 \
    --num-layers 8 \
    --learning-rate 1e-4

echo ""
echo "=== Training Complete ==="
echo "Adapter saved to: $OUTPUT_DIR/adapter"

# Create adapter_config.json
cat > "$OUTPUT_DIR/adapter_config.json" << 'CONFIG'
{
  "name": "base-adapter-with-tests",
  "adapter_type": "base",
  "base_model": "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
  "patterns": [
    "error-handling",
    "null-safety",
    "type-safety",
    "async-await",
    "validation",
    "test-structure",
    "test-assertion",
    "test-setup",
    "test-mock"
  ],
  "languages": ["typescript"],
  "version": "1.0.0",
  "description": "Base adapter with common patterns and test patterns"
}
CONFIG

echo "Config saved to: $OUTPUT_DIR/adapter_config.json"
