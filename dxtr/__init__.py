from pathlib import Path
from contextvars import ContextVar
from functools import wraps
import asyncio
import os
import time
from pydantic_ai_litellm import LiteLLMModel
from dxtr import data_models

# === Shared LLM Config (lazy initialization) ===
# Models are created on first access to avoid failing at import time.
# This allows tests and local dev to import the module without LiteLLM configured.

_models: dict[str, LiteLLMModel] = {}


def _get_litellm_config() -> tuple[str, str]:
    """Get LiteLLM config, raising helpful error if not configured."""
    base_url = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")
    api_key = os.environ.get("LITELLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LITELLM_API_KEY not set. See .env.example for required configuration."
        )
    return base_url, api_key


def _get_model(name: str) -> LiteLLMModel:
    """Get or create a LiteLLM model (lazy initialization)."""
    if name not in _models:
        base_url, api_key = _get_litellm_config()
        _models[name] = LiteLLMModel(f"openai/{name}", api_base=base_url, api_key=api_key)
    return _models[name]


# Lazy model accessors - these are module-level properties that create models on first use
class _LazyModel:
    """Descriptor that creates LiteLLM model on first access."""

    def __init__(self, name: str):
        self.name = name

    def __get__(self, obj, objtype=None) -> LiteLLMModel:
        return _get_model(self.name)


class _Models:
    """Container for lazy-loaded models. Access via module-level vars."""

    master = _LazyModel("master")
    github_summarizer = _LazyModel("github_summarizer")
    profile_synthesizer = _LazyModel("profile_synthesizer")
    papers_ranker = _LazyModel("papers_ranker")


_lazy_models = _Models()


# Module-level accessors for backwards compatibility
# These are accessed like `from dxtr import master` and lazily create models
def __getattr__(name: str):
    """Module-level __getattr__ for lazy model access."""
    if name in ("master", "github_summarizer", "profile_synthesizer", "papers_ranker"):
        return _get_model(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# === Session Context ===
_session_id: ContextVar[str | None] = ContextVar("session_id", default=None)
_session_tags: ContextVar[list[str]] = ContextVar("session_tags", default=[])
_session_metadata: ContextVar[dict] = ContextVar("session_metadata", default={})
_session_state: ContextVar[data_models.SessionState | None] = ContextVar(
    "session_state", default=None
)


def set_session_id(session_id: str) -> None:
    """Set the current session ID for LiteLLM/Langfuse tracing."""
    _session_id.set(session_id)


def set_session_tags(tags: list[str]) -> None:
    """Set tags for Langfuse filtering (e.g., ['test', 'integration'])."""
    _session_tags.set(tags)


def set_session_metadata(metadata: dict) -> None:
    """Set additional metadata for Langfuse (e.g., {'scenario': 'profile_creation'})."""
    _session_metadata.set(metadata)


def set_session_state(state: data_models.SessionState) -> None:
    """Set the current user's session state (loaded from GCS at turn start)."""
    _session_state.set(state)


def get_session_state() -> data_models.SessionState:
    """Get the current session state. Returns empty state if not set."""
    state = _session_state.get()
    if state is None:
        return data_models.SessionState()
    return state


def update_session_state(**kwargs) -> None:
    """Update specific fields in the current session state.

    Called by tools after creating artifacts to keep state fresh within a turn.
    """
    current = get_session_state()
    updated = current.model_copy(update=kwargs)
    _session_state.set(updated)


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


async def run_agent(agent, query: str, deps, **kwargs):
    """Run an agent with a prompt.

    Args:
        agent: The pydantic-ai Agent to run
        query: The prompt/query string
        deps: Dependencies for the agent (must match agent's deps_type, or None if no deps_type)
        **kwargs: Additional args passed to agent.run() (e.g., model_settings)

    NOTE: Streaming is disabled because pydantic-ai's run_stream() doesn't properly
    execute tools - it returns text before tool execution completes. This is a known
    issue with pydantic-ai streaming when models return text before tool calls.
    """
    return await agent.run(query, deps=deps, **kwargs)


def log_tool_usage(func):
    """Decorator that logs when a tool function is called.

    Works with both sync and async functions. Sends status to internal bus.

    Usage:
        @agent.tool_plain
        @log_tool_usage
        async def my_tool(request: MyRequest) -> str:
            ...
    """
    from dxtr.bus import send_internal

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        send_internal("tool", f"{func.__name__} started.")
        _t = time.time()
        ret = await func(*args, **kwargs)
        send_internal("tool", f"{func.__name__} finished, {time.time() - _t:.1f}s elapsed.")
        return ret

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        send_internal("tool", f"{func.__name__} called")
        return func(*args, **kwargs)

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper
