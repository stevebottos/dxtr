from pathlib import Path
from contextvars import ContextVar

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


DXTR_DIR = Path.home() / ".dxtr"
DXTR_DIR.mkdir(parents=True, exist_ok=True)

# === Shared LLM Config ===
LITELLM_BASE_URL = "http://localhost:4000/v1"
LITELLM_API_KEY = "sk-1234"

provider = OpenAIProvider(base_url=LITELLM_BASE_URL, api_key=LITELLM_API_KEY)
master = OpenAIChatModel("master", provider=provider)
github_summarizer = OpenAIChatModel("github_summarizer", provider=provider)
profile_synthesizer = OpenAIChatModel("profile_synthesizer", provider=provider)


# === Session Context ===
_session_id: ContextVar[str | None] = ContextVar("session_id", default=None)


def set_session_id(session_id: str) -> None:
    """Set the current session ID for LiteLLM tracing."""
    _session_id.set(session_id)


def get_model_settings() -> dict:
    """Get model_settings with current session metadata for LiteLLM."""
    session_id = _session_id.get()
    if session_id:
        settings = {"extra_body": {"litellm_session_id": session_id}}
        print(f"  [model_settings] litellm_session_id={session_id}")
        return settings
    print("  [model_settings] WARNING: No session_id set!")
    return {}


def load_system_prompt(file_path: Path) -> str:
    """Load a system prompt from a markdown file."""
    return file_path.read_text().strip()
