"""End-to-end pipeline for generating transformation training data.

This module provides a complete pipeline for:
1. Extracting code transformations from git repositories
2. Classifying and filtering for learnable examples
3. Formatting as Alpaca training data
4. Creating train/valid splits

Can be used as a module or run as a CLI script.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from .diff_extractor import CodeTransformPair, GitDiffExtractor
from .pattern_classifier import ClassificationResult, PatternClassifier
from .transform_formatter import (
    TransformAlpacaFormatter,
    TransformExample,
    create_train_valid_split,
)

logger = logging.getLogger(__name__)


class TransformDataPipeline:
    """End-to-end pipeline for generating transformation training data.

    Integrates:
    - GitDiffExtractor: Extract before/after code pairs from git history
    - PatternClassifier: Classify and filter for learnable examples
    - TransformAlpacaFormatter: Format as training data

    Usage:
        pipeline = TransformDataPipeline([Path("/path/to/repo")])
        stats = pipeline.generate(
            transform_types=["error-handling", "null-safety"],
            output_dir=Path("data/transforms")
        )
    """

    def __init__(
        self,
        repo_paths: list[Path],
        use_llm_classification: bool = False,
        llm_client: Any | None = None,
    ):
        """Initialize pipeline.

        Args:
            repo_paths: List of git repository paths to extract from
            use_llm_classification: Whether to use LLM for classification validation
            llm_client: LLM client instance (required if use_llm_classification=True)
        """
        self.repo_paths = [Path(p).resolve() for p in repo_paths]
        self.classifier = PatternClassifier(llm_client=llm_client)
        self.formatter = TransformAlpacaFormatter()
        self.use_llm = use_llm_classification

        # Validate repos exist
        for path in self.repo_paths:
            if not (path / ".git").exists():
                logger.warning(f"Not a git repository: {path}")

    def generate(
        self,
        transform_types: list[str] | None = None,
        max_examples_per_type: int = 500,
        max_commits_per_repo: int = 1000,
        output_dir: Path | str = Path("data/transforms"),
        file_patterns: list[str] | None = None,
        train_ratio: float = 0.9,
        output_format: str = "text",
    ) -> dict[str, Any]:
        """Generate training data for code transformations.

        Args:
            transform_types: Types to extract (default: all types)
            max_examples_per_type: Maximum examples per transform type
            max_commits_per_repo: Maximum commits to analyze per repo
            output_dir: Output directory for generated data
            file_patterns: File patterns to include (default: ["*.ts", "*.tsx"])
            train_ratio: Train/validation split ratio
            output_format: Output format ("text" for mlx-lm, "alpaca" for standard)

        Returns:
            Dictionary with statistics:
            {
                "total_extracted": int,
                "total_learnable": int,
                "by_type": {"error-handling": {"extracted": N, "learnable": M}, ...},
                "train_examples": int,
                "valid_examples": int,
                "output_dir": str
            }
        """
        if file_patterns is None:
            file_patterns = ["*.ts", "*.tsx"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Extract transformation pairs from all repos
        logger.info("Step 1: Extracting transformation pairs from git history...")
        all_pairs: dict[str, list[CodeTransformPair]] = defaultdict(list)

        for repo_path in self.repo_paths:
            if not (repo_path / ".git").exists():
                continue

            try:
                extractor = GitDiffExtractor(repo_path)
                pairs = extractor.extract_transforms(
                    file_patterns=file_patterns,
                    max_commits=max_commits_per_repo,
                    transform_types=transform_types,
                )

                for pair in pairs:
                    all_pairs[pair.transform_type].append(pair)

                logger.info(f"  Extracted {len(pairs)} pairs from {repo_path.name}")
            except Exception as e:
                logger.error(f"  Failed to process {repo_path}: {e}")

        # Step 2: Classify and filter for learnable examples
        logger.info("Step 2: Classifying and filtering learnable examples...")
        learnable_pairs: list[tuple[CodeTransformPair, ClassificationResult]] = []
        stats_by_type: dict[str, dict[str, int]] = {}

        for transform_type, pairs in all_pairs.items():
            # Limit per type
            pairs = pairs[:max_examples_per_type * 2]  # Extract more, filter down

            # Classify each pair
            type_learnable: list[tuple[CodeTransformPair, ClassificationResult]] = []
            for pair in pairs:
                result = self.classifier.classify(pair, use_llm=self.use_llm)
                if result.is_learnable:
                    type_learnable.append((pair, result))

                    if len(type_learnable) >= max_examples_per_type:
                        break

            learnable_pairs.extend(type_learnable)

            stats_by_type[transform_type] = {
                "extracted": len(pairs),
                "learnable": len(type_learnable),
            }

            logger.info(
                f"  {transform_type}: {len(type_learnable)}/{len(pairs)} learnable"
            )

        # Step 3: Format as training examples
        logger.info("Step 3: Formatting as Alpaca training examples...")
        examples = self.formatter.format_batch(learnable_pairs)

        # Step 4: Create train/valid split and save
        logger.info("Step 4: Creating train/valid split and saving...")
        train_examples, valid_examples = create_train_valid_split(
            examples, train_ratio=train_ratio
        )

        train_path = output_dir / "train.jsonl"
        valid_path = output_dir / "valid.jsonl"

        self.formatter.to_jsonl(train_examples, train_path, format=output_format)
        self.formatter.to_jsonl(valid_examples, valid_path, format=output_format)

        # Also save full dataset as JSON for inspection
        self.formatter.to_json(
            examples, output_dir / "all_examples.json", include_metadata=True
        )

        logger.info(f"  Saved {len(train_examples)} train, {len(valid_examples)} valid")
        logger.info(f"  Output directory: {output_dir}")

        return {
            "total_extracted": sum(len(pairs) for pairs in all_pairs.values()),
            "total_learnable": len(learnable_pairs),
            "by_type": stats_by_type,
            "train_examples": len(train_examples),
            "valid_examples": len(valid_examples),
            "output_dir": str(output_dir),
        }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate code transformation training data from git repositories"
    )
    parser.add_argument(
        "--repos",
        nargs="+",
        type=Path,
        required=True,
        help="Paths to git repositories",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["error-handling", "null-safety", "type-safety", "async-await", "validation"],
        help="Types of transformations to extract (default: all)",
    )
    parser.add_argument(
        "--max-per-type",
        type=int,
        default=500,
        help="Maximum examples per transform type (default: 500)",
    )
    parser.add_argument(
        "--max-commits",
        type=int,
        default=1000,
        help="Maximum commits to analyze per repo (default: 1000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/transforms"),
        help="Output directory (default: data/transforms)",
    )
    parser.add_argument(
        "--file-patterns",
        nargs="+",
        default=["*.ts", "*.tsx"],
        help="File patterns to include (default: *.ts *.tsx)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Train/validation split ratio (default: 0.9)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "alpaca"],
        default="text",
        help="Output format (default: text for mlx-lm)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Run pipeline
    pipeline = TransformDataPipeline(args.repos)
    stats = pipeline.generate(
        transform_types=args.types,
        max_examples_per_type=args.max_per_type,
        max_commits_per_repo=args.max_commits,
        output_dir=args.output,
        file_patterns=args.file_patterns,
        train_ratio=args.train_ratio,
        output_format=args.format,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Transformation Data Generation Complete")
    print("=" * 60)
    print(f"\nTotal extracted:  {stats['total_extracted']}")
    print(f"Total learnable:  {stats['total_learnable']}")
    print(f"Train examples:   {stats['train_examples']}")
    print(f"Valid examples:   {stats['valid_examples']}")
    print(f"\nBy type:")
    for transform_type, type_stats in stats["by_type"].items():
        print(f"  {transform_type}: {type_stats['learnable']}/{type_stats['extracted']}")
    print(f"\nOutput: {stats['output_dir']}")


if __name__ == "__main__":
    main()
