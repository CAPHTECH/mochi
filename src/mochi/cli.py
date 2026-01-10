"""CLI entry point for Mochi."""

from pathlib import Path

import click


@click.group()
@click.version_option()
def main() -> None:
    """Mochi - Domain-specific SLM generator for software projects."""
    pass


@main.command()
@click.option("--repo", "-r", required=True, help="Git repository URL or local path")
@click.option("--output", "-o", default="./data", help="Output directory for training data")
@click.option("--extensions", "-e", multiple=True, default=[".ts", ".tsx"], help="File extensions")
@click.option("--project-name", "-n", default="project", help="Project name for context")
def prepare(repo: str, output: str, extensions: tuple[str, ...], project_name: str) -> None:
    """Prepare training data from a repository."""
    from mochi.data_generation.alpaca_converter import create_training_dataset
    from mochi.ingestion.git_connector import GitConnector
    from mochi.preprocessing.code_chunker import ChunkStrategy, CodeChunker

    click.echo(f"Preparing data from: {repo}")

    # Clone or connect to repo
    if repo.startswith(("http://", "https://", "git@")):
        target_path = Path(output) / "repo"
        click.echo(f"Cloning to: {target_path}")
        connector = GitConnector.clone(repo, target_path)
    else:
        connector = GitConnector(repo)

    # Get source files
    files = connector.get_source_files(list(extensions))
    click.echo(f"Found {len(files)} source files")

    # Chunk code
    chunker = CodeChunker()
    all_chunks = []

    with click.progressbar(files, label="Chunking files") as bar:
        for source_file in bar:
            chunks = chunker.chunk(
                source_file.path,
                source_file.content,
                source_file.language,
                strategy=ChunkStrategy.TOPLEVEL,
            )
            all_chunks.extend(chunks)

    click.echo(f"Created {len(all_chunks)} chunks")

    # Convert to training format
    output_dir = Path(output)
    train_path, eval_path = create_training_dataset(
        all_chunks,
        output_dir,
        project_name=project_name,
    )

    click.echo(f"Training data saved to: {train_path}")
    click.echo(f"Evaluation data saved to: {eval_path}")


@main.command()
@click.option("--train", "-t", required=True, help="Training data file (JSONL)")
@click.option("--eval", "-e", default=None, help="Evaluation data file (JSONL)")
@click.option("--output", "-o", default="./output", help="Output directory")
@click.option("--base-model", "-m", default="Qwen/Qwen2.5-Coder-1.5B", help="Base model")
@click.option("--epochs", default=2, help="Number of training epochs")
@click.option("--batch-size", default=4, help="Batch size")
@click.option("--learning-rate", default=2e-4, help="Learning rate")
def train(
    train: str,
    eval: str | None,
    output: str,
    base_model: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    """Train a domain-specific SLM using QLoRA."""
    from mochi.training.trainer import MochiTrainer, QLoRAConfig

    click.echo(f"Training with base model: {base_model}")

    config = QLoRAConfig(
        base_model=base_model,
        output_dir=output,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    trainer = MochiTrainer(config)
    click.echo("Setting up model...")
    trainer.setup()

    click.echo("Starting training...")
    adapter_path = trainer.train(train, eval)

    click.echo(f"Adapter saved to: {adapter_path}")


@main.command()
@click.option("--base-model", "-m", default="Qwen/Qwen2.5-Coder-1.5B", help="Base model")
@click.option("--finetuned", "-f", default=None, help="Fine-tuned model path")
@click.option("--eval-file", "-e", required=True, help="Evaluation file (JSONL)")
@click.option("--output", "-o", default="./eval_results.json", help="Output file")
@click.option("--max-samples", default=50, help="Maximum samples to evaluate")
def evaluate(
    base_model: str,
    finetuned: str | None,
    eval_file: str,
    output: str,
    max_samples: int,
) -> None:
    """Evaluate and compare models."""
    from mochi.training.evaluator import ModelEvaluator, compare_outputs

    click.echo("Loading models...")
    evaluator = ModelEvaluator(base_model, finetuned)
    evaluator.setup()

    click.echo(f"Evaluating {max_samples} samples...")
    results = evaluator.evaluate_file(eval_file, output, max_samples)

    stats = compare_outputs(results)
    click.echo("\nResults:")
    click.echo(f"  Total samples: {stats['total']}")
    click.echo(f"  Base model avg output length: {stats['base_avg_length']:.1f}")
    if finetuned:
        click.echo(f"  Fine-tuned avg output length: {stats['finetuned_avg_length']:.1f}")

    click.echo(f"\nDetailed results saved to: {output}")


@main.command()
@click.option("--adapter", "-a", required=True, help="LoRA adapter path")
@click.option("--base-model", "-m", default="Qwen/Qwen2.5-Coder-1.5B", help="Base model")
@click.option("--output", "-o", required=True, help="Output path for merged model")
def merge(adapter: str, base_model: str, output: str) -> None:
    """Merge LoRA adapter with base model."""
    from mochi.training.trainer import MochiTrainer, QLoRAConfig

    click.echo(f"Merging adapter: {adapter}")

    config = QLoRAConfig(base_model=base_model)
    trainer = MochiTrainer(config)
    trainer.setup()

    trainer.merge_and_save(adapter, output)
    click.echo(f"Merged model saved to: {output}")


if __name__ == "__main__":
    main()
