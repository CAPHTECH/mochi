#!/usr/bin/env python3
"""Regenerate training data with LSP context.

This script regenerates the training data for kiri project with
LSP-based context extraction for improved API/schema name accuracy.

Improvements in v2:
- Filters global builtin functions from LSP context
- Adds receiver type labels (e.g., "// Methods on DuckDBClient:")
- Includes negative examples for hallucination prevention
- Increased method call examples (3 -> 10)
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mochi.data_generation.alpaca_converter import create_training_dataset_with_context
from mochi.data_generation.negative_examples import NegativeExampleGenerator
from mochi.data_generation.yaml_examples import generate_yaml_training_data
from mochi.ingestion.git_connector import GitConnector
from mochi.preprocessing.code_chunker import ChunkStrategy, CodeChunker


async def regenerate_with_lsp_context():
    """Regenerate training data with LSP context extraction."""

    project_root = Path(__file__).parent.parent
    repo_path = project_root / "data" / "repo"
    output_dir = project_root / "data" / "mlx_lsp"

    print("=" * 70)
    print("Regenerating training data with LSP context")
    print("=" * 70)
    print(f"Repository: {repo_path}")
    print(f"Output: {output_dir}")
    print()

    if not repo_path.exists():
        print(f"ERROR: Repository not found at {repo_path}")
        return False

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Connect to repository
    print("Connecting to repository...")
    connector = GitConnector(str(repo_path))

    # Get TypeScript files
    extensions = [".ts", ".tsx"]
    files = connector.get_source_files(extensions)
    print(f"Found {len(files)} source files")

    # Chunk code
    print("\nChunking files...")
    chunker = CodeChunker()
    all_chunks = []

    for source_file in files:
        try:
            chunks = chunker.chunk(
                source_file.path,
                source_file.content,
                source_file.language,
                strategy=ChunkStrategy.TOPLEVEL,
            )
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  Warning: Failed to chunk {source_file.path}: {e}")

    print(f"Created {len(all_chunks)} chunks")

    # Generate training data with LSP context
    print("\nGenerating training data with LSP context...")
    print("(This may take a while as LSP extracts context for each chunk)")

    try:
        train_path, eval_path = await create_training_dataset_with_context(
            chunks=all_chunks,
            output_dir=output_dir,
            project_root=repo_path,
            project_name="kiri",
            language="typescript",
            train_ratio=0.9,
        )

        print(f"\nTraining data saved to: {train_path}")
        print(f"Evaluation data saved to: {eval_path}")

        # Add negative examples for hallucination prevention
        # Increased counts to improve context compliance and reduce synonym usage
        print("\nGenerating negative examples for hallucination prevention...")
        neg_generator = NegativeExampleGenerator(include_kiri_specific=True)
        negative_examples = neg_generator.generate_all(
            correction_count=150,
            identification_count=75,
            constrained_count=150,
            exact_match_count=200,  # NEW: Emphasize exact method names
            rejection_count=100,    # NEW: Teach rejection of synonyms
        )
        print(f"Generated {len(negative_examples)} negative examples")

        # Append negative examples to training data
        with open(train_path, "a", encoding="utf-8") as f:
            for example in negative_examples:
                if example.input:
                    text = (
                        f"### Instruction:\n{example.instruction}\n\n"
                        f"### Input:\n{example.input}\n\n"
                        f"### Response:\n{example.output}"
                    )
                else:
                    text = (
                        f"### Instruction:\n{example.instruction}\n\n"
                        f"### Response:\n{example.output}"
                    )
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

        print(f"Appended negative examples to {train_path}")

        # P1/P3: Generate YAML/config file examples (enhanced)
        print("\nGenerating YAML config examples (P3 enhanced)...")
        yaml_examples = generate_yaml_training_data(
            yaml_dir=repo_path,
            project_name="kiri",
            completion_count=100,
            key_count=50,
            value_count=50,
            comment_count=50,   # P3.2: コメント活用例
            array_count=50,     # P3.4: 配列アイテム補完
            dataset_count=100,  # P3.3: データセットYAML
            use_nested=True,    # P3.1: ネスト構造対応
        )
        print(f"Generated {len(yaml_examples)} YAML config examples (target: 400+)")

        # Append YAML examples to training data
        if yaml_examples:
            with open(train_path, "a", encoding="utf-8") as f:
                for example in yaml_examples:
                    if example.input:
                        text = (
                            f"### Instruction:\n{example.instruction}\n\n"
                            f"### Input:\n{example.input}\n\n"
                            f"### Response:\n{example.output}"
                        )
                    else:
                        text = (
                            f"### Instruction:\n{example.instruction}\n\n"
                            f"### Response:\n{example.output}"
                        )
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            print(f"Appended YAML examples to {train_path}")

        # Count total examples
        with open(train_path) as f:
            total_lines = sum(1 for _ in f)
        print(f"\nTotal training examples: {total_lines}")

        # Show sample
        print("\n--- Sample training example ---")
        with open(train_path) as f:
            first_line = f.readline()
            example = json.loads(first_line)

        # Show just the input portion which should have context
        input_text = example.get("text", "")[:500]
        print(input_text)
        if len(example.get("text", "")) > 500:
            print("...")

        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(regenerate_with_lsp_context())
    sys.exit(0 if success else 1)
