#!/usr/bin/env python3
"""Detailed LSP context extraction test with kiri's DuckDB module."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mochi.lsp import LSPClient, create_context_extractor


async def test_duckdb_context():
    """Test LSP context extraction on DuckDB client module."""

    project_root = Path(__file__).parent.parent / "data" / "repo"
    test_file = project_root / "src" / "shared" / "duckdb.ts"

    if not test_file.exists():
        print(f"ERROR: Test file not found: {test_file}")
        return False

    print(f"Testing context extraction on: {test_file.relative_to(project_root)}")
    print("=" * 70)

    # Read file and find interesting positions
    content = test_file.read_text()
    lines = content.split("\n")

    # Find positions with 'this.' method calls
    test_positions = []
    for i, line in enumerate(lines):
        if "this." in line:
            idx = line.find("this.")
            # Position after 'this.'
            test_positions.append((i, idx + 5, line.strip()[:60]))
        if "connection." in line:
            idx = line.find("connection.")
            test_positions.append((i, idx + 11, line.strip()[:60]))

    print(f"\nFound {len(test_positions)} test positions with method calls")

    try:
        print("\nStarting LSP server...")
        extractor = await create_context_extractor(
            project_root=project_root,
            language="typescript",
        )
        print("LSP server ready.\n")

        # Test a few positions
        for line, col, context in test_positions[:5]:
            print(f"--- Line {line + 1}: {context} ---")
            print(f"    Cursor at column {col}")

            ctx = await extractor.extract_at_position(test_file, line, col)

            if ctx.is_empty():
                print("    Result: (no context)")
            else:
                if ctx.methods:
                    print(f"    Methods ({len(ctx.methods)}): {', '.join(ctx.methods[:5])}")
                if ctx.types:
                    print(f"    Types ({len(ctx.types)}): {', '.join(ctx.types[:5])}")
            print()

        # Get hover info at a specific position
        print("--- Testing Hover Info ---")
        # Find 'DuckDBClient' class definition
        for i, line in enumerate(lines):
            if "class DuckDBClient" in line:
                print(f"Getting hover at 'DuckDBClient' (line {i + 1})")
                hover = await extractor.lsp.get_hover(test_file, i, line.find("DuckDBClient") + 1)
                if hover:
                    print(f"    Hover: {hover.contents[:200]}...")
                break

        # Show workspace symbols relevant to DuckDB
        print("\n--- DuckDB-related Symbols ---")
        symbols = await extractor.lsp.get_workspace_symbols("DuckDB")
        for s in symbols[:10]:
            print(f"  - {s.name} ({s.kind.name})")

        await extractor.lsp.stop()
        print("\n" + "=" * 70)
        print("Detailed test completed!")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_duckdb_context())
    sys.exit(0 if success else 1)
