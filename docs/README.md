# Mochi - Domain-Specific Code Completion

A library that improves code completion accuracy using adapters trained on project-specific patterns.

## Features

- **Base Adapter + Project Adapter** - Two-layer architecture with common patterns and project-specific patterns
- **Apple Silicon Optimized** - Fast inference via MLX (M1/M2/M3/M4 support)
- **MCP Server** - Integration with Claude Code/Claude Desktop
- **Lightweight** - Runs on 0.5B-3B parameter models
- **Private Project Support** - Train and distribute adapters for private codebases

## Supported Models

| Model | Preset | Memory | Code Gen | Diff Gen | Pattern |
|-------|--------|--------|----------|----------|---------|
| **Qwen3-Coder-30B 4-bit** | `qwen3-coder` | 64GB | 5/5 | 5/5 | 4/5 |
| GPT-OSS-20B 8-bit | `gpt-oss` | 128GB | 3/5 | 1/5 | 2/5 |

**Recommendation**: Use `qwen3-coder` (default) for production. Better code generation and diff output.

## Installation

### From GitHub (Current)

```bash
# Basic installation
pip install git+https://github.com/CAPHTECH/mochi.git

# For Apple Silicon (recommended)
pip install "mochi[mlx] @ git+https://github.com/CAPHTECH/mochi.git"

# Full installation with training capabilities
pip install "mochi[mlx,training] @ git+https://github.com/CAPHTECH/mochi.git"
```

### From PyPI (After Publication)

```bash
# Basic installation
pip install mochi

# For Apple Silicon (recommended)
pip install mochi[mlx]

# Full installation with training capabilities
pip install mochi[mlx,training]
```

### For Development

```bash
git clone https://github.com/CAPHTECH/mochi.git
cd mochi
pip install -e ".[dev]"
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

### Quick Distribution with `mochi pack` / `mochi install`

```bash
# 1. Package trained adapter
mochi pack ./output/my-project-adapter --name my-project
# Creates: my-project.mochi.tar.gz (~500MB-3GB depending on model)

# 2. Share package (upload to S3, shared drive, etc.)
aws s3 cp my-project.mochi.tar.gz s3://my-bucket/adapters/

# 3. Team members install
mochi install s3://my-bucket/adapters/my-project.mochi.tar.gz
# or from local file:
mochi install ./my-project.mochi.tar.gz

# 4. Start server (auto-detects installed adapter)
mochi serve
```

**.mochi package structure:**
```
my-project.mochi/
├── manifest.json           # Package metadata (name, model, version)
├── adapter_config.json     # Adapter configuration
└── adapters.safetensors    # LoRA weights
```

### Storage Options

| Storage Option | Use Case | Notes |
|----------------|----------|-------|
| **Local path** | Single developer | `~/.mochi/adapters/` |
| **S3/GCS/Azure** | Team/Enterprise | Private bucket with IAM |
| **Git LFS** | Version controlled | In project repo |
| **HuggingFace Hub** | Public adapters | For shared base adapters |

**Note:** Base model (~16GB) is automatically downloaded by mlx_lm on first run. Only the adapter package needs to be distributed.

## CLI Commands

| Command | Description |
|---------|-------------|
| `mochi init` | Initialize project |
| `mochi prepare` | Prepare training data |
| `mochi train base` | Train Base Adapter |
| `mochi train project` | Train Project Adapter |
| `mochi pack` | Package adapter for distribution (.mochi.tar.gz) |
| `mochi install` | Install adapter from URL or local file |
| `mochi serve` | Start MCP server (auto-detects installed adapters) |
| `mochi list` | List installed adapters |

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
- Memory:
  - Qwen3-Coder-30B 4-bit: 64GB+ recommended
  - GPT-OSS-20B 8-bit: 128GB+ recommended

## Tech Stack

- **Inference**: MLX (Apple Silicon) / PyTorch (CUDA)
- **Base Models**:
  - Qwen3-Coder-30B-A3B 4-bit MoE (recommended)
  - GPT-OSS-20B 8-bit
- **Training**: LoRA/QLoRA
- **Integration**: MCP (Model Context Protocol)

## License

MIT License
