# Mochi - Domain-Specific Code Completion

A library that improves code completion accuracy using adapters trained on project-specific patterns.

## Features

- **Base Adapter + Project Adapter** - Two-layer architecture with common patterns and project-specific patterns
- **Apple Silicon Optimized** - Fast inference via MLX (M1/M2/M3/M4 support)
- **MCP Server** - Integration with Claude Code/Claude Desktop
- **Lightweight** - Runs on 0.5B-3B parameter models
- **Private Project Support** - Train and distribute adapters for private codebases

## Installation

```bash
# Basic installation
pip install mochi

# For Apple Silicon (recommended)
pip install mochi[mlx]

# Full installation with training capabilities
pip install mochi[mlx,training]
```

## Quick Start

### 1. Use as MCP Server with uvx (Recommended)

Add to Claude Code configuration (`~/.claude/settings.json` or project `.claude/settings.json`):

```json
{
  "mcpServers": {
    "mochi": {
      "command": "uvx",
      "args": [
        "--from", "mochi[mlx]",
        "mochi", "serve",
        "--base", "~/.mochi/adapters/my-project"
      ]
    }
  }
}
```

### 2. Private Project Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  1. Train adapter locally                                    │
│     mochi train project --data ./data --output ./adapter     │
│                                                              │
│  2. Upload to private storage                                │
│     - S3 / GCS / Azure Blob                                 │
│     - Private Git repo (Git LFS)                            │
│     - Shared network drive                                  │
│                                                              │
│  3. Each team member downloads adapter                       │
│     ~/.mochi/adapters/my-project/                           │
│                                                              │
│  4. Configure Claude Code MCP                                │
│     Point to local adapter path                             │
└─────────────────────────────────────────────────────────────┘
```

**Train adapter for your project:**

```bash
# Initialize project
mochi init --project /path/to/your-project

# Prepare training data
mochi prepare --repo /path/to/your-project --output ./data/my-project

# Train Project Adapter
mochi train project \
  --data ./data/my-project \
  --output ./adapter \
  --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
```

**Distribute adapter to team:**

```bash
# Upload to S3 (example)
aws s3 sync ./adapter s3://my-bucket/mochi-adapters/my-project/

# Team members download
aws s3 sync s3://my-bucket/mochi-adapters/my-project/ ~/.mochi/adapters/my-project/
```

### 3. Use Directly from Python

```python
from mochi.adapters import BaseAdapter, ProjectAdapter, AdapterStack
from mochi.inference import InferenceEngine

# Load adapters
base = BaseAdapter.load("./adapters/base-ts-v1")
project = ProjectAdapter.load("./adapters/my-project", base_adapter=base)

# Configure stack
stack = AdapterStack([
    (base, 0.3),      # Common patterns: 30%
    (project, 0.7),   # Project-specific: 70%
])

# Inference
engine = InferenceEngine(adapter_stack=stack)
result = engine.complete(
    instruction="Fill in the code",
    input_code="const users = await db.",
)
print(result.response)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      mochi Library                           │
├─────────────────────────────────────────────────────────────┤
│  Base Adapter (Pre-trained)                                  │
│  - error-handling, async/await patterns                      │
│  - type-safety, validation patterns                          │
├─────────────────────────────────────────────────────────────┤
│  Project Adapter (Project-specific)                          │
│  - Naming conventions and coding style                       │
│  - Custom API patterns and error handling                    │
├─────────────────────────────────────────────────────────────┤
│  MCP Server                                                  │
│  - Claude Code/Desktop integration                           │
│  - domain_query, complete_code, generate_diff                │
└─────────────────────────────────────────────────────────────┘
```

## Adapter Distribution

| Storage Option | Use Case | Notes |
|----------------|----------|-------|
| **Local path** | Single developer | `~/.mochi/adapters/` |
| **S3/GCS/Azure** | Team/Enterprise | Private bucket with IAM |
| **Git LFS** | Version controlled | In project repo |
| **HuggingFace Hub** | Public adapters | For shared base adapters |

**Adapter files (~50MB):**
```
adapter/
├── adapter_config.json      # Metadata
├── adapters.safetensors     # LoRA weights
└── tokenizer files...       # Tokenizer config
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `mochi init` | Initialize project |
| `mochi prepare` | Prepare training data |
| `mochi train base` | Train Base Adapter |
| `mochi train project` | Train Project Adapter |
| `mochi serve` | Start MCP server |
| `mochi list` | List available adapters |

## MCP Tools

| Tool | Description |
|------|-------------|
| `domain_query` | Domain-specific code generation (modes: auto/conservative/creative) |
| `complete_code` | Code completion (prefix/suffix support) |
| `generate_diff` | Generate unified diff |
| `suggest_pattern` | Pattern suggestions |

## Claude Code MCP Configuration Examples

**With uvx (recommended):**
```json
{
  "mcpServers": {
    "mochi": {
      "command": "uvx",
      "args": ["--from", "mochi[mlx]", "mochi", "serve", "--base", "~/.mochi/adapters/my-project"]
    }
  }
}
```

**With pip-installed mochi:**
```json
{
  "mcpServers": {
    "mochi": {
      "command": "mochi",
      "args": ["serve", "--base", "~/.mochi/adapters/my-project"]
    }
  }
}
```

**With specific Python:**
```json
{
  "mcpServers": {
    "mochi": {
      "command": "/opt/homebrew/bin/python3.11",
      "args": ["-m", "mochi.cli.main", "serve", "--base", "~/.mochi/adapters/my-project"]
    }
  }
}
```

## Requirements

- Python 3.11+
- macOS (Apple Silicon recommended) / Linux / Windows
- Memory: 8GB+ recommended

## Tech Stack

- **Inference**: MLX (Apple Silicon) / PyTorch (CUDA)
- **Base Model**: Qwen3-Coder-30B-A3B (4-bit MoE)
- **Training**: LoRA/QLoRA
- **Integration**: MCP (Model Context Protocol)

## License

MIT License
