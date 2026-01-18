from pathlib import Path
from contextvars import ContextVar
import os

from pydantic_ai_litellm import LiteLLMModel


DXTR_DIR = Path.home() / ".dxtr"
DXTR_DIR.mkdir(parents=True, exist_ok=True)

# Debug mode (default True unless DXTR_PROD=true)
DEBUG_MODE = os.environ.get("DXTR_PROD", "false").lower() != "true"

# === Shared LLM Config ===
LITELLM_BASE_URL = "http://localhost:4000"
LITELLM_API_KEY = "sk-1234"

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
        output = await stream.get_output()

    print(f"\n{'='*60}\n")
    return StreamResult(output, stream)
