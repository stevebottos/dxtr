"""
Message buses for agent communication.

Two buses:
- internal_bus: Agent-to-agent (tool status, progress, debugging)
- user_bus: Agent-to-user (rankings, direct content that bypasses LLM)

Architecture:
    User ←→ Master Agent ←→ Tools/Subagents
                ↓                  ↓
           user_bus          internal_bus
                ↓                  ↓
           Final answer      SSE status events

Why queues instead of context variables?
Pydantic-ai executes tools in isolated contexts. ContextVar changes in tools
don't propagate back to the caller. But shared queue objects work because
the reference is copied, not the contents.

Usage:
    # At request start (server):
    internal_q, user_q = setup_buses()

    # In a tool:
    from dxtr.bus import send_to_user, send_internal
    send_to_user(rankings_text)  # Goes directly to user
    send_internal("tool", "Ranking complete")  # Status event

    # At request end (server):
    user_content = collect_user_content(user_q)
    teardown_buses()
"""

import asyncio
from contextvars import ContextVar

_internal_queue: ContextVar[asyncio.Queue | None] = ContextVar("internal_queue", default=None)
_user_queue: ContextVar[asyncio.Queue | None] = ContextVar("user_queue", default=None)


def setup_buses(maxsize: int = 100) -> tuple[asyncio.Queue, asyncio.Queue]:
    """Create both buses for a request. Call at start of request.

    Returns:
        Tuple of (internal_queue, user_queue)
    """
    internal = asyncio.Queue(maxsize=maxsize)
    user = asyncio.Queue(maxsize=maxsize)
    _internal_queue.set(internal)
    _user_queue.set(user)
    return internal, user


def teardown_buses() -> None:
    """Clear both buses. Call at end of request."""
    _internal_queue.set(None)
    _user_queue.set(None)


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


def send_to_user(content: str) -> None:
    """Send content directly to the user.

    Use for content that must be preserved exactly (rankings, formatted output).
    Bypasses the LLM so formatting is guaranteed.
    """
    queue = _user_queue.get()
    if queue is not None:
        try:
            queue.put_nowait(content)
        except asyncio.QueueFull:
            print("[WARN] User bus full, dropping content", flush=True)


def collect_user_content(queue: asyncio.Queue) -> list[str]:
    """Collect all content from the user bus.

    Call after agent completes to get all user-directed content.
    """
    content = []
    while not queue.empty():
        try:
            content.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    return content
