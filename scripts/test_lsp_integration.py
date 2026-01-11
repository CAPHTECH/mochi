#!/usr/bin/env python3
"""Integration test for LSP context extraction with kiri project."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mochi.lsp import LSPClient, ContextExtractor, create_context_extractor


async def test_lsp_with_kiri():
    """Test LSP context extraction with actual kiri project."""

    project_root = Path(__file__).parent.parent / "data" / "repo"

    if not project_root.exists():
        print(f"ERROR: kiri project not found at {project_root}")
        return False

    print(f"Testing LSP with kiri project at: {project_root}")
    print("=" * 60)

    # Find a TypeScript file to test
    ts_files = list(project_root.glob("src/**/*.ts"))
    if not ts_files:
        print("ERROR: No TypeScript files found in src/")
        return False

    test_file = ts_files[0]
    print(f"\nTest file: {test_file.relative_to(project_root)}")

    # Read file content to find a good test position
    content = test_file.read_text()
    lines = content.split("\n")
    print(f"File has {len(lines)} lines")

    # Find a line with a dot (method access)
    test_positions = []
    for i, line in enumerate(lines):
        if "." in line and not line.strip().startswith("//"):
            dot_pos = line.find(".")
            test_positions.append((i, dot_pos + 1))
            if len(test_positions) >= 3:
                break

    if not test_positions:
        print("WARNING: No suitable test positions found, using line 10")
        test_positions = [(min(10, len(lines) - 1), 0)]

    print(f"Test positions: {test_positions}")
    print()

    # Test LSP connection
    print("Starting LSP server...")
    try:
        extractor = await create_context_extractor(
            project_root=project_root,
            language="typescript",
        )
        print("LSP server started successfully!")

        # Extract context at test positions
        for line, col in test_positions:
            print(f"\n--- Position ({line}, {col}) ---")
            print(f"Line content: {lines[line][:80]}...")

            try:
                context = await extractor.extract_at_position(test_file, line, col)

                if context.is_empty():
                    print("  Context: (empty)")
                else:
                    formatted = context.format()
                    print("  Context:")
                    for ctx_line in formatted.split("\n"):
                        print(f"    {ctx_line}")

                    # Count items
                    print(f"  Methods: {len(context.methods)}")
                    print(f"  Types: {len(context.types)}")
                    print(f"  Imports: {len(context.imports)}")

            except Exception as e:
                print(f"  ERROR: {e}")

        # Also get workspace symbols
        print("\n--- Workspace Symbols (sample) ---")
        try:
            symbols = await extractor.lsp.get_workspace_symbols("")
            print(f"Total symbols: {len(symbols)}")
            for symbol in symbols[:10]:
                print(f"  - {symbol.name} ({symbol.kind.name})")
            if len(symbols) > 10:
                print(f"  ... and {len(symbols) - 10} more")
        except Exception as e:
            print(f"  ERROR getting symbols: {e}")

        await extractor.lsp.stop()
        print("\n" + "=" * 60)
        print("Integration test PASSED!")
        return True

    except Exception as e:
        print(f"ERROR: LSP connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_lsp_with_kiri())
    sys.exit(0 if success else 1)
