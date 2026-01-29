from pathlib import Path
from contextvars import ContextVar
from functools import wraps
import asyncio
import os
import time
from pydantic_ai_litellm import LiteLLMModel
from dxtr import data_models

# === Shared LLM Config ===
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY")

# Models via LiteLLM proxy
master = LiteLLMModel("openai/master", api_base=LITELLM_BASE_URL, api_key=LITELLM_API_KEY)
github_summarizer = LiteLLMModel(
    "openai/github_summarizer", api_base=LITELLM_BASE_URL, api_key=LITELLM_API_KEY
)
profile_synthesizer = LiteLLMModel(
    "openai/profile_synthesizer", api_base=LITELLM_BASE_URL, api_key=LITELLM_API_KEY
)
papers_ranker = LiteLLMModel(
    "openai/papers_ranker", api_base=LITELLM_BASE_URL, api_key=LITELLM_API_KEY
)


# === Session Context ===
_session_id: ContextVar[str | None] = ContextVar("session_id", default=None)
_session_tags: ContextVar[list[str]] = ContextVar("session_tags", default=[])
_session_metadata: ContextVar[dict] = ContextVar("session_metadata", default={})


def set_session_id(session_id: str) -> None:
    """Set the current session ID for LiteLLM/Langfuse tracing."""
    _session_id.set(session_id)


def set_session_tags(tags: list[str]) -> None:
    """Set tags for Langfuse filtering (e.g., ['test', 'integration'])."""
    _session_tags.set(tags)


def set_session_metadata(metadata: dict) -> None:
    """Set additional metadata for Langfuse (e.g., {'scenario': 'profile_creation'})."""
    _session_metadata.set(metadata)


def get_model_settings() -> dict:
    """Get model_settings with current session metadata for LiteLLM/Langfuse."""
    session_id = _session_id.get()
    tags = _session_tags.get()
    metadata = _session_metadata.get()

    if not session_id and not tags and not metadata:
        return {}

    extra_body = {}
    langfuse_metadata = {}

    if session_id:
        extra_body["litellm_session_id"] = session_id
        langfuse_metadata["session_id"] = session_id

    if tags:
        langfuse_metadata["tags"] = tags

    if metadata:
        langfuse_metadata.update(metadata)

    if langfuse_metadata:
        extra_body["metadata"] = langfuse_metadata

    return {"extra_body": extra_body}


# === Event Bus ===
# Per-request queue for streaming events to clients
_event_queue: ContextVar[asyncio.Queue | None] = ContextVar("event_queue", default=None)


def create_event_queue(maxsize: int = 100) -> asyncio.Queue:
    """Create and set a new event queue for the current request context."""
    queue = asyncio.Queue(maxsize=maxsize)
    _event_queue.set(queue)
    return queue


def get_event_queue() -> asyncio.Queue | None:
    """Get the event queue for the current context (if any)."""
    return _event_queue.get()


def clear_event_queue() -> None:
    """Clear the event queue from the current context."""
    _event_queue.set(None)


def publish(event_type: str, message: str) -> None:
    """Publish an event to the bus.

    Args:
        event_type: Event type (e.g., "tool", "progress", "status", "error")
        message: Human-readable message

    Events are added to the current request's queue (if one exists) and
    always printed to stdout for server logs.
    """
    # Always log to stdout
    print(f"[{event_type.upper()}] {message}", flush=True)

    # Push to queue if one exists for this request
    queue = _event_queue.get()
    if queue is not None:
        try:
            queue.put_nowait({"type": event_type, "message": message})
        except asyncio.QueueFull:
            print(f"[WARN] Event queue full, dropping: {event_type}", flush=True)


def load_system_prompt(file_path: Path) -> str:
    """Load a system prompt from a markdown file."""
    return file_path.read_text().strip()


class StreamResult:
    """Wrapper to make streaming result compatible with AgentRunResult interface."""

    def __init__(self, output, stream):
        self.output = output
        self._stream = stream

    def all_messages(self):
        return self._stream.all_messages()


async def run_agent(agent, query: data_models.MasterRequest, deps, **kwargs):
    """Run an agent.

    NOTE: Streaming is disabled because pydantic-ai's run_stream() doesn't properly
    execute tools - it returns text before tool execution completes. This is a known
    issue with pydantic-ai streaming when models return text before tool calls.
    See: https://github.com/pydantic/pydantic-ai/issues (streaming + tool_calls)
    """
    # Always use non-streaming to ensure tools are executed properly
    return await agent.run(query, deps=deps, **kwargs)


def log_tool_usage(func):
    """Decorator that logs when a tool function is called.

    Works with both sync and async functions. Publishes to the event bus
    when the tool is invoked to improve visibility.

    Usage:
        @agent.tool_plain
        @log_tool_usage
        async def my_tool(request: MyRequest) -> str:
            ...
    """

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        publish("tool", f"{func.__name__} started.")
        _t = time.time()
        ret = await func(*args, **kwargs)
        publish("tool", f"{func.__name__} finished, {time.time() - _t} elapsed.")
        return ret

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        publish("tool", f"{func.__name__} called")
        return func(*args, **kwargs)

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper
