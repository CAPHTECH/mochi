"""Async utility functions for mochi library.

Provides helper functions for bridging sync and async code.
"""

from __future__ import annotations

import asyncio
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine in a synchronous context.

    Creates a new event loop, runs the coroutine to completion,
    and properly cleans up the loop. This is useful when async
    code needs to be called from synchronous code.

    Note: This should not be called from within an existing event loop.
    Use `await` directly in async contexts instead.

    Args:
        coro: Async coroutine to execute

    Returns:
        The result of the coroutine

    Raises:
        Any exception raised by the coroutine

    Example:
        async def fetch_data():
            return await some_async_operation()

        # Call from sync code
        result = run_sync(fetch_data())
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
