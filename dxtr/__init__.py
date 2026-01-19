from pathlib import Path
from contextvars import ContextVar
from functools import wraps
import asyncio
import os

from pydantic_ai_litellm import LiteLLMModel


DXTR_DIR = Path.home() / ".dxtr"
DXTR_DIR.mkdir(parents=True, exist_ok=True)

# Debug mode (default True unless DXTR_PROD=true)
DEBUG_MODE = os.environ.get("DXTR_PROD", "false").lower() != "true"

# === Shared LLM Config ===
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "sk-1234")

# Models via LiteLLM proxy
master = LiteLLMModel("openai/master", api_base=LITELLM_BASE_URL, api_key=LITELLM_API_KEY)
github_summarizer = LiteLLMModel("openai/github_summarizer", api_base=LITELLM_BASE_URL, api_key=LITELLM_API_KEY)
profile_synthesizer = LiteLLMModel("openai/profile_synthesizer", api_base=LITELLM_BASE_URL, api_key=LITELLM_API_KEY)
papers_ranker = LiteLLMModel("openai/papers_ranker", api_base=LITELLM_BASE_URL, api_key=LITELLM_API_KEY)


# === Session Context ===
_session_id: ContextVar[str | None] = ContextVar("session_id", default=None)


def set_session_id(session_id: str) -> None:
    """Set the current session ID for LiteLLM tracing."""
    _session_id.set(session_id)


def get_model_settings() -> dict:
    """Get model_settings with current session metadata for LiteLLM."""
    session_id = _session_id.get()
    if session_id:
        return {"extra_body": {"litellm_session_id": session_id}}
    return {}


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


async def run_agent(agent, prompt: str, **kwargs):
    """Run an agent, streaming to console in debug mode."""
    if not DEBUG_MODE:
        return await agent.run(prompt, **kwargs)

    # Debug: stream output to console
    print(f"\n{'='*60}")
    print(f"[STREAM] {agent.name or 'agent'}")
    print(f"{'='*60}", flush=True)

    async with agent.run_stream(prompt, **kwargs) as stream:
        async for text in stream.stream_text(delta=True):
            print(text, end="", flush=True)
            publish("text", text)
        output = await stream.get_output()

    print(f"\n{'='*60}\n")
    return StreamResult(output, stream)


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
        publish("tool", f"{func.__name__} called")
        return await func(*args, **kwargs)

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        publish("tool", f"{func.__name__} called")
        return func(*args, **kwargs)

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def requires(*tool_names: str):
    """Decorator that documents prerequisite tools in the docstring.

    The model will see "PREREQUISITE: Call X first" in the tool description,
    guiding it to check state before calling this tool.

    Usage:
        @agent.tool_plain
        @requires("get_papers")
        async def fetch_and_download_papers(...):
            ...
    """
    def decorator(func):
        prereq_list = ", ".join(tool_names)
        prereq_text = f"\n\nPREREQUISITE: Call {prereq_list} first."

        # Augment docstring
        original_doc = func.__doc__ or ""
        func.__doc__ = original_doc.rstrip() + prereq_text

        return func
    return decorator
