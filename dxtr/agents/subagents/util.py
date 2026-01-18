"""Shared utilities for subagents."""

import asyncio
from typing import TypeVar, Callable, Awaitable

from dxtr import publish

T = TypeVar("T")
R = TypeVar("R")


async def parallel_map(
    items: list[T],
    func: Callable[[T, int, int], Awaitable[R]],
    desc: str = "Processing",
    status_interval: float = 10.0,
    max_concurrency: int | None = None,
    on_progress: Callable[[int, int, R], None] | None = None,
) -> list[R]:
    """Execute async function on items in parallel with progress tracking.

    Args:
        items: List of items to process
        func: Async function that takes (item, index, total) and returns result.
              Index is 1-based. Function should handle its own exceptions.
        desc: Description for progress messages
        status_interval: Seconds between background status updates (0 to disable)
        max_concurrency: Max concurrent tasks (None for unlimited)
        on_progress: Optional callback(completed, total, result) called after each item

    Returns:
        List of results in same order as items

    Example:
        async def score_paper(paper: dict, idx: int, total: int) -> dict:
            print(f"  [{idx}/{total}] Scoring: {paper['title'][:40]}")
            result = await agent.run(...)
            return {"id": paper["id"], "score": result.output.score}

        results = await parallel_map(
            papers,
            score_paper,
            desc="Ranking papers",
            on_progress=lambda done, total, r: print(f"Progress: {done}/{total}"),
        )
    """
    total = len(items)
    if total == 0:
        return []

    completed_count = 0
    pending_indices: set[int] = set(range(total))
    results: list[R | None] = [None] * total
    lock = asyncio.Lock()

    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def process_one(idx: int, item: T) -> None:
        nonlocal completed_count

        async def _run() -> None:
            nonlocal completed_count
            result = await func(item, idx + 1, total)  # 1-based index
            results[idx] = result

            async with lock:
                pending_indices.discard(idx)
                completed_count += 1
                current_completed = completed_count

            if on_progress:
                on_progress(current_completed, total, result)

        if semaphore:
            async with semaphore:
                await _run()
        else:
            await _run()

    async def print_pending_status() -> None:
        """Background task to print status periodically."""
        while True:
            await asyncio.sleep(status_interval)
            async with lock:
                remaining = len(pending_indices)
                done = completed_count
            if remaining == 0:
                break
            publish("progress", f"{desc}: {done}/{total} done, {remaining} pending")

    publish("status", f"{desc}: {total} items...")

    tasks = [process_one(idx, item) for idx, item in enumerate(items)]

    status_task = None
    if status_interval > 0:
        status_task = asyncio.create_task(print_pending_status())

    try:
        await asyncio.gather(*tasks)
    finally:
        if status_task:
            status_task.cancel()
            try:
                await status_task
            except asyncio.CancelledError:
                pass

    return results  # type: ignore


class ProgressReporter:
    """Helper for subagents to report progress back to callers.

    This provides a structured way for subagents to send updates during
    long-running operations. Uses the event bus by default.

    Example:
        reporter = ProgressReporter()
        reporter.update("Starting analysis...")
        reporter.progress(1, 10, "Processed file 1")
        reporter.complete("Analysis finished")
    """

    def update(self, message: str) -> None:
        """Send a general status update."""
        publish("status", message)

    def progress(self, completed: int, total: int, detail: str = "") -> None:
        """Send a progress update with completion count."""
        pct = (completed / total * 100) if total > 0 else 0
        msg = f"[{completed}/{total}] ({pct:.0f}%)"
        if detail:
            msg += f" {detail}"
        publish("progress", msg)

    def complete(self, message: str = "Done") -> None:
        """Send a completion message."""
        publish("status", f"✓ {message}")

    def error(self, message: str) -> None:
        """Send an error message."""
        publish("error", message)
