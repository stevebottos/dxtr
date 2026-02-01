"""
Internal message bus for agent status events.

Used for streaming SSE status updates (tool started, progress, etc.) to the frontend
while the agent is working.

Why a queue instead of context variables?
Pydantic-ai executes tools in isolated contexts. ContextVar changes in tools
don't propagate back to the caller. But shared queue objects work because
the reference is copied, not the contents.

Usage:
    # At request start (server):
    queue = setup_bus()

    # In a tool:
    from dxtr.bus import send_internal
    send_internal("tool", "Processing...")

    # At request end (server):
    teardown_bus()
"""

import asyncio
from contextvars import ContextVar

_internal_queue: ContextVar[asyncio.Queue | None] = ContextVar("internal_queue", default=None)


def setup_bus(maxsize: int = 100) -> asyncio.Queue:
    """Create the internal bus for a request. Call at start of request."""
    queue = asyncio.Queue(maxsize=maxsize)
    _internal_queue.set(queue)
    return queue


def teardown_bus() -> None:
    """Clear the bus. Call at end of request."""
    _internal_queue.set(None)


def send_internal(event_type: str, message: str) -> None:
    """Send a status event on the internal bus.

    Use for tool status, progress updates, debugging info.
    These get streamed to the frontend as SSE events.
    """
    queue = _internal_queue.get()
    if queue is not None:
        try:
            queue.put_nowait({"type": event_type, "message": message})
        except asyncio.QueueFull:
            print(f"[WARN] Internal bus full, dropping: {event_type}", flush=True)
    # Always log to stdout for server logs
    print(f"[{event_type.upper()}] {message}", flush=True)
