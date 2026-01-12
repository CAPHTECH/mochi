"""CLI entry point for mochi library.

Provides commands for:
- init: Initialize a project for mochi
- train: Train base or project adapters
- pack: Package adapter for distribution
- install: Install adapter from URL or local file
- serve: Start MCP server for Claude Code integration
- list: List installed adapters
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
    default="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
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
    default="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
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


def _expand_adapter_path(path: str | None) -> Path | None:
    """Expand ~ and resolve adapter path."""
    if path is None:
        return None
    # Expand ~ to home directory
    expanded = Path(path).expanduser()
    # If it's a relative path, resolve it
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return expanded.resolve()


@main.command()
@click.argument("adapter_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    help="Output path for package",
)
@click.option(
    "--name", "-n",
    help="Package name (default: directory name)",
)
@click.option(
    "--description", "-d",
    default="",
    help="Package description",
)
@click.option(
    "--no-compress",
    is_flag=True,
    help="Create uncompressed directory package instead of .tar.gz",
)
def pack(
    adapter_dir: Path,
    output: Path | None,
    name: str | None,
    description: str,
    no_compress: bool,
) -> None:
    """Package an adapter for distribution.

    Creates a .mochi.tar.gz archive that can be shared with team members.

    \b
    Examples:
      # Pack adapter
      mochi pack output/my-project-adapter

      # Pack with custom name
      mochi pack output/adapter --name my-project --output ./dist/

      # Create uncompressed package
      mochi pack output/adapter --no-compress
    """
    from ..packaging import pack_adapter

    click.echo(f"Packing adapter: {adapter_dir}")

    try:
        package_path = pack_adapter(
            adapter_dir=adapter_dir,
            output_path=output,
            name=name,
            description=description,
            compress=not no_compress,
        )
        click.echo(f"Package created: {package_path}")
        click.echo(f"Size: {package_path.stat().st_size / 1024 / 1024:.1f} MB")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@main.command()
@click.argument("source")
@click.option(
    "--name", "-n",
    help="Override adapter name",
)
@click.option(
    "--target", "-t",
    type=click.Path(path_type=Path),
    help="Installation directory (default: ~/.mochi/adapters/)",
)
def install(source: str, name: str | None, target: Path | None) -> None:
    """Install an adapter from URL or local file.

    SOURCE can be:
    - Local file path: ./my-adapter.mochi.tar.gz
    - HTTP URL: https://example.com/adapter.mochi.tar.gz
    - S3 URL: s3://bucket/path/adapter.mochi.tar.gz

    The base model will be automatically downloaded on first use.

    \b
    Examples:
      # Install from local file
      mochi install ./my-project.mochi.tar.gz

      # Install from URL
      mochi install https://example.com/adapters/my-project.mochi.tar.gz

      # Install from S3
      mochi install s3://my-bucket/adapters/my-project.mochi.tar.gz

      # Install with custom name
      mochi install ./adapter.tar.gz --name my-project
    """
    from ..packaging import install_package

    click.echo(f"Installing adapter from: {source}")

    try:
        install_path = install_package(
            source=source,
            target_dir=target,
            name=name,
        )
        click.echo(f"Installed to: {install_path}")

        # Show manifest info
        from ..packaging import MochiPackage
        package = MochiPackage(install_path)
        manifest = package.manifest

        click.echo(f"\nAdapter: {manifest.name}")
        click.echo(f"Type: {manifest.adapter_type}")
        click.echo(f"Base model: {manifest.base_model}")
        if manifest.description:
            click.echo(f"Description: {manifest.description}")

        click.echo(f"\nTo use with Claude Code, add to settings.json:")
        click.echo(f'  "mochi": {{"command": "mochi", "args": ["serve", "--adapter", "{install_path}"]}}')

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@main.command()
@click.option(
    "--adapter", "-a",
    type=str,
    help="Adapter path (supports ~) - auto-detects from ~/.mochi/adapters/ if not specified",
)
@click.option(
    "--base", "-b",
    type=str,
    help="Base adapter path (supports ~) or HuggingFace repo ID",
)
@click.option(
    "--project", "-p",
    type=str,
    help="Project adapter path (supports ~)",
)
@click.option(
    "--model", "-m",
    default="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
    help="Base model name (auto-downloaded if not cached)",
)
@click.option(
    "--preset",
    type=click.Choice(["qwen3-coder", "qwen3-coder-base", "gpt-oss"]),
    help="Use preset configuration",
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
    adapter: str | None,
    base: str | None,
    project: str | None,
    model: str,
    preset: str | None,
    base_weight: float,
    project_weight: float,
) -> None:
    """Start MCP server for Claude Code integration.

    The server provides code completion tools using trained adapters.
    Base model is automatically downloaded on first run (~16GB).

    If no adapter is specified, looks for installed adapters in ~/.mochi/adapters/.

    \b
    Examples:
      # Auto-detect installed adapter
      mochi serve

      # Use specific adapter
      mochi serve --adapter ~/.mochi/adapters/my-project

      # Use preset
      mochi serve --preset qwen3-coder
    """
    from ..serving import start_server

    click.echo("Starting mochi MCP server...", err=True)

    # Determine adapter path
    adapter_path = None
    if adapter:
        adapter_path = _expand_adapter_path(adapter)
    elif not base and not project and not preset:
        # Auto-detect from installed adapters
        from ..packaging import get_default_adapter
        adapter_path = get_default_adapter()
        if adapter_path:
            click.echo(f"Auto-detected adapter: {adapter_path}", err=True)
        else:
            click.echo("No adapters installed. Using base model only.", err=True)
            click.echo("Install an adapter with: mochi install <package>", err=True)

    # Expand paths for base/project
    base_path = _expand_adapter_path(base) if base and "/" in base and not base.startswith(("http", "hf:")) else None
    project_path = _expand_adapter_path(project) if project else None

    # For HuggingFace repo IDs, pass as string
    base_adapter_arg = base_path or base
    project_adapter_arg = project_path or project

    # If adapter specified via --adapter, use it as project adapter
    if adapter_path and not project_adapter_arg:
        project_adapter_arg = adapter_path

    if base_adapter_arg:
        click.echo(f"Base Adapter: {base_adapter_arg}", err=True)
    if project_adapter_arg:
        click.echo(f"Project Adapter: {project_adapter_arg}", err=True)
    if preset:
        click.echo(f"Preset: {preset}", err=True)

    click.echo(f"Model: {model}", err=True)
    click.echo("Note: Model will be downloaded automatically if not cached (~16GB)", err=True)
    click.echo(f"Weights: base={base_weight}, project={project_weight}", err=True)
    click.echo("Server ready. Waiting for JSON-RPC requests on stdin...", err=True)

    # Start the MCP server
    start_server(
        base_adapter=base_adapter_arg,
        project_adapter=project_adapter_arg,
        base_model=model,
        base_weight=base_weight,
        project_weight=project_weight,
        preset=preset,
    )


@main.command("list")
@click.option(
    "--adapters-dir", "-a",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Adapters directory to scan (default: ~/.mochi/adapters/)",
)
def list_adapters(adapters_dir: Path | None) -> None:
    """List installed adapters."""
    from ..packaging import list_installed_adapters, DEFAULT_ADAPTERS_DIR

    target_dir = adapters_dir or DEFAULT_ADAPTERS_DIR
    adapters = list_installed_adapters(target_dir)

    if not adapters:
        click.echo(f"No adapters found in {target_dir}")
        click.echo("\nInstall an adapter with:")
        click.echo("  mochi install <package-url>")
        return

    click.echo(f"Installed adapters ({target_dir}):\n")

    for adapter in adapters:
        click.echo(f"  {adapter['name']}")
        click.echo(f"    Type: {adapter['type']}")
        click.echo(f"    Model: {adapter['base_model']}")
        click.echo(f"    Path: {adapter['path']}")
        if adapter.get('description'):
            click.echo(f"    Description: {adapter['description']}")
        click.echo()


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
    click.echo(f"Validation data saved to: {eval_path}")


if __name__ == "__main__":
    main()
