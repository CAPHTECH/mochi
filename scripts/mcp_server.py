#!/usr/bin/env python3
"""MCP Server entry point for Mochi.

Usage:
    # Run with MLX backend (recommended for Apple Silicon)
    python scripts/mcp_server.py --backend mlx --preset qwen3-coder

    # Run with GPT-OSS model
    python scripts/mcp_server.py --backend mlx --preset gpt-oss

    # Run with PyTorch backend (for CUDA or custom models)
    python scripts/mcp_server.py --backend pytorch --adapter output/adapter

    # Claude Code configuration (claude_code_config.json):
    {
        "mcpServers": {
            "mochi": {
                "command": "python",
                "args": ["scripts/mcp_server.py", "--backend", "mlx", "--preset", "qwen3-coder"]
            }
        }
    }

Available presets (MLX backend):
    - qwen3-coder: Qwen3-Coder-30B-A3B (fast, high quality)
    - gpt-oss: GPT-OSS-20B (OpenAI's open model)
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mochi.mcp.server import MochiMCPServer, ServerConfig


def main():
    parser = argparse.ArgumentParser(
        description="Mochi MCP Server - Domain-specific SLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # MLX with Qwen3-Coder (recommended)
  %(prog)s --backend mlx --preset qwen3-coder

  # MLX with GPT-OSS
  %(prog)s --backend mlx --preset gpt-oss

  # PyTorch with custom adapter
  %(prog)s --backend pytorch --base-model Qwen/Qwen3-Coder-30B-A3B --adapter output/adapter
        """,
    )

    # Backend selection
    parser.add_argument(
        "--backend",
        type=str,
        choices=["mlx", "pytorch"],
        default="mlx",
        help="Inference backend: mlx (Apple Silicon) or pytorch (default: mlx)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["qwen3-coder", "gpt-oss"],
        default="qwen3-coder",
        help="Model preset for MLX backend (default: qwen3-coder)",
    )

    # Custom model settings
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Custom base model ID (overrides preset)",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Custom path to LoRA adapter (overrides preset)",
    )

    # Runtime settings
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Inference timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--max-memory",
        type=float,
        default=64.0,
        help="Maximum memory usage in GB (default: 64)",
    )

    # Resource directories
    parser.add_argument(
        "--patterns-dir",
        type=str,
        default=None,
        help="Directory containing pattern markdown files",
    )
    parser.add_argument(
        "--conventions-dir",
        type=str,
        default=None,
        help="Directory containing convention markdown files",
    )

    args = parser.parse_args()

    # Build config
    config = ServerConfig(
        backend=args.backend,
        preset=args.preset if args.backend == "mlx" else None,
        base_model=args.base_model or "Qwen/Qwen3-Coder-30B-A3B",
        adapter_path=args.adapter,
        timeout_seconds=args.timeout,
        max_memory_gb=args.max_memory,
        patterns_dir=args.patterns_dir,
        conventions_dir=args.conventions_dir,
    )

    # Log startup info to stderr (stdout is for JSON-RPC)
    print(f"Starting Mochi MCP Server...", file=sys.stderr)
    print(f"  Backend: {config.backend}", file=sys.stderr)
    if config.preset:
        print(f"  Preset: {config.preset}", file=sys.stderr)
    print(f"  Timeout: {config.timeout_seconds}s", file=sys.stderr)
    print(file=sys.stderr)

    server = MochiMCPServer(config)

    try:
        server.run_stdio()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
