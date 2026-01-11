"""CLI entry point for mochi library.

Provides commands for:
- init: Initialize a project for mochi
- train: Train base or project adapters
- serve: Start MCP server for Claude Code integration
"""

from __future__ import annotations

from pathlib import Path

import click


@click.group()
@click.version_option()
def main() -> None:
    """mochi - Domain-specific SLM for code completion.

    mochi provides Base Adapters (common patterns) and Project Adapters
    (project-specific patterns) for enhanced code completion.

    Examples:

      # Initialize project
      mochi init --project /path/to/project

      # Train base adapter
      mochi train base --data data/common-patterns/ --output output/base/

      # Train project adapter
      mochi train project --base output/base/ --from-lsp /path/to/project

      # Start MCP server
      mochi serve --base output/base/ --project output/project/
    """
    pass


@main.command()
@click.option(
    "--project", "-p",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=Path.cwd(),
    help="Project directory (default: current directory)",
)
def init(project: Path) -> None:
    """Initialize a project for mochi.

    Creates configuration files and directory structure for mochi adapters.
    """
    from ..core.config import MochiConfig

    project = project.resolve()
    click.echo(f"Initializing mochi for: {project}")

    # Create .mochi directory
    mochi_dir = project / ".mochi"
    mochi_dir.mkdir(exist_ok=True)

    # Create adapters directory
    adapters_dir = mochi_dir / "adapters"
    adapters_dir.mkdir(exist_ok=True)

    # Create data directory
    data_dir = mochi_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # Create default config
    config = MochiConfig(
        adapters_dir=adapters_dir,
        output_dir=mochi_dir / "output",
        data_dir=data_dir,
    )
    config.save(mochi_dir / "config.yaml")

    click.echo(f"Created .mochi/ directory structure")
    click.echo(f"Configuration saved to: {mochi_dir / 'config.yaml'}")


@main.group()
def train() -> None:
    """Train adapters (base or project)."""
    pass


@train.command("base")
@click.option(
    "--data", "-d",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Training data directory (must contain train.jsonl)",
)
@click.option(
    "--output", "-o",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path("output/base-adapter"),
    help="Output directory for adapter",
)
@click.option(
    "--model", "-m",
    default="mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    help="Base model name",
)
@click.option(
    "--name", "-n",
    default="mochi-base",
    help="Name for the adapter",
)
@click.option(
    "--iters",
    type=int,
    default=200,
    help="Training iterations",
)
@click.option(
    "--batch-size",
    type=int,
    default=4,
    help="Batch size",
)
@click.option(
    "--learning-rate",
    type=float,
    default=1e-5,
    help="Learning rate",
)
def train_base(
    data: Path,
    output: Path,
    model: str,
    name: str,
    iters: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    """Train a Base Adapter on common patterns.

    Base Adapters are trained on common TypeScript/JavaScript patterns:
    - error-handling (try-catch, Result types)
    - async-await (Promise handling)
    - type-safety (type annotations, guards)
    - null-safety (optional chaining)
    - validation (zod, assertions)
    """
    from ..adapters.base_adapter import BaseAdapter
    from ..core.types import TrainingConfig

    click.echo(f"Training Base Adapter: {name}")
    click.echo(f"  Model: {model}")
    click.echo(f"  Data: {data}")
    click.echo(f"  Output: {output}")

    training_config = TrainingConfig(
        base_model=model,
        epochs=iters // 100 + 1,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=output,
    )

    try:
        adapter = BaseAdapter.train(
            name=name,
            base_model=model,
            training_data=data,
            output_dir=output,
            training_config=training_config,
        )
        click.echo(f"\nBase Adapter saved to: {adapter.adapter_path}")

    except Exception as e:
        click.echo(f"Training failed: {e}", err=True)
        raise click.Abort()


@train.command("project")
@click.option(
    "--base", "-b",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Base adapter directory (optional)",
)
@click.option(
    "--data", "-d",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Training data directory (must contain train.jsonl)",
)
@click.option(
    "--from-lsp",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Extract training data from project using LSP",
)
@click.option(
    "--output", "-o",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path("output/project-adapter"),
    help="Output directory for adapter",
)
@click.option(
    "--model", "-m",
    default="mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    help="Base model name",
)
@click.option(
    "--name", "-n",
    help="Name for the adapter (defaults to project name)",
)
@click.option(
    "--iters",
    type=int,
    default=200,
    help="Training iterations",
)
def train_project(
    base: Path | None,
    data: Path | None,
    from_lsp: Path | None,
    output: Path,
    model: str,
    name: str | None,
    iters: int,
) -> None:
    """Train a Project Adapter on project-specific patterns.

    Project Adapters are fine-tuned on:
    - Custom types and interfaces from LSP
    - Project naming conventions
    - Custom error classes
    - Domain-specific utilities

    Either --data or --from-lsp must be provided.
    """
    from ..adapters.base_adapter import BaseAdapter
    from ..adapters.project_adapter import ProjectAdapter
    from ..core.types import TrainingConfig

    if data is None and from_lsp is None:
        click.echo("Error: Either --data or --from-lsp must be provided", err=True)
        raise click.Abort()

    # Load base adapter if provided
    base_adapter = None
    if base:
        click.echo(f"Loading Base Adapter from: {base}")
        base_adapter = BaseAdapter.load(base)

    if from_lsp:
        # Create from LSP
        project_root = from_lsp.resolve()
        adapter_name = name or project_root.name

        click.echo(f"Creating Project Adapter from LSP: {adapter_name}")
        click.echo(f"  Project: {project_root}")
        click.echo(f"  Output: {output}")

        try:
            adapter = ProjectAdapter.from_lsp(
                project_root=project_root,
                name=adapter_name,
                base_adapter=base_adapter,
                base_model=model,
                output_dir=output,
            )
            click.echo(f"\nProject Adapter saved to: {adapter.adapter_path}")

        except Exception as e:
            click.echo(f"Training failed: {e}", err=True)
            raise click.Abort()

    else:
        # Train from data
        adapter_name = name or "project-adapter"

        click.echo(f"Training Project Adapter: {adapter_name}")
        click.echo(f"  Model: {model}")
        click.echo(f"  Data: {data}")
        click.echo(f"  Output: {output}")

        training_config = TrainingConfig(
            base_model=model,
            epochs=iters // 100 + 1,
            output_dir=output,
        )

        try:
            adapter = ProjectAdapter.train(
                name=adapter_name,
                base_model=model,
                training_data=data,  # type: ignore
                output_dir=output,
                base_adapter=base_adapter,
                training_config=training_config,
            )
            click.echo(f"\nProject Adapter saved to: {adapter.adapter_path}")

        except Exception as e:
            click.echo(f"Training failed: {e}", err=True)
            raise click.Abort()


@main.command()
@click.option(
    "--base", "-b",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Base adapter directory",
)
@click.option(
    "--project", "-p",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Project adapter directory",
)
@click.option(
    "--model", "-m",
    default="mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    help="Base model name (if no adapters specified)",
)
@click.option(
    "--base-weight",
    type=float,
    default=0.3,
    help="Weight for base adapter (0.0-1.0)",
)
@click.option(
    "--project-weight",
    type=float,
    default=0.7,
    help="Weight for project adapter (0.0-1.0)",
)
def serve(
    base: Path | None,
    project: Path | None,
    model: str,
    base_weight: float,
    project_weight: float,
) -> None:
    """Start MCP server for Claude Code integration.

    The server provides the domain_query tool for code completion
    using the specified adapters.

    The server communicates over stdio using JSON-RPC 2.0 protocol.
    Configure Claude Code to use this server via mcp_servers config.
    """
    from ..serving import MCPServer, start_server
    from ..serving.mcp_server import MCPServerConfig

    click.echo("Starting mochi MCP server...", err=True)

    if base:
        click.echo(f"Base Adapter: {base}", err=True)
    if project:
        click.echo(f"Project Adapter: {project}", err=True)
    if not base and not project:
        click.echo("Warning: No adapters specified. Using base model only.", err=True)

    click.echo(f"Model: {model}", err=True)
    click.echo(f"Weights: base={base_weight}, project={project_weight}", err=True)
    click.echo("Server ready. Waiting for JSON-RPC requests on stdin...", err=True)

    # Start the MCP server
    start_server(
        base_adapter=base,
        project_adapter=project,
        base_model=model,
        base_weight=base_weight,
        project_weight=project_weight,
    )


@main.command()
@click.option(
    "--adapters-dir", "-a",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=Path("adapters"),
    help="Adapters directory to scan",
)
def list(adapters_dir: Path) -> None:
    """List available adapters."""
    from ..adapters.registry import AdapterRegistry

    registry = AdapterRegistry(adapters_dir)
    count = registry.discover_all()

    if count == 0:
        click.echo(f"No adapters found in {adapters_dir}")
        return

    click.echo(f"Found {count} adapters:\n")

    # List base adapters
    base_adapters = registry.list_base_adapters()
    if base_adapters:
        click.echo("Base Adapters:")
        for name, info in base_adapters.items():
            patterns = ", ".join(info.get("patterns", []))
            click.echo(f"  {name}: {patterns}")

    # List project adapters
    project_adapters = registry.list_project_adapters()
    if project_adapters:
        click.echo("\nProject Adapters:")
        for name, info in project_adapters.items():
            base = info.get("base_adapter", "none")
            click.echo(f"  {name} (base: {base})")


# Legacy commands from old CLI (for backward compatibility)
@main.command()
@click.option("--repo", "-r", required=True, help="Git repository URL or local path")
@click.option("--output", "-o", default="./data", help="Output directory for training data")
@click.option("--extensions", "-e", multiple=True, default=[".ts", ".tsx"], help="File extensions")
@click.option("--project-name", "-n", default="project", help="Project name for context")
def prepare(repo: str, output: str, extensions: tuple[str, ...], project_name: str) -> None:
    """Prepare training data from a repository (legacy command)."""
    from ..data_generation.alpaca_converter import create_training_dataset
    from ..ingestion.git_connector import GitConnector
    from ..preprocessing.code_chunker import ChunkStrategy, CodeChunker

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


if __name__ == "__main__":
    main()
