"""
Base Agent for SGLang-based agents.

Provides common functionality for prompt loading and SGLang backend management.
"""

from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import sglang as sgl

from dxtr.config_v2 import config

@dataclass
class GlobalState:
    """Simple global state tracking for the agent."""
    profile_loaded: bool = False
    profile_required: bool = True
    github_summary_loaded: bool = False
    github_summary_required: bool = False
    today_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

    def check_state(self):
        """Update state based on file existence."""
        self.profile_loaded = config.paths.profile_file.exists()
        self.github_summary_loaded = config.paths.github_summary_file.exists()
        self.today_date = datetime.now().strftime("%Y-%m-%d")

class BaseAgent:
    """Base class for SGLang-powered agents."""

    def __init__(self):
        """Initialize SGLang agent with backend connection.

        Args:
            prompts_dir: Directory containing agent's prompt files.
                        Defaults to 'prompts' subdirectory of agent module.
        """
        # Initialize and check global state
        self.state = GlobalState()
        self.state.check_state()

        # Connect to SGLang backend
        raw_url = config.sglang.base_url
        base_url = raw_url.replace("/v1", "").rstrip("/")

        try:
            self.backend = sgl.RuntimeEndpoint(base_url)
            sgl.set_default_backend(self.backend)
        except Exception as e:
            print(f"Connection failed to SGLang at {base_url}")
            raise e

    @staticmethod
    def load_system_prompt(path: Path | str) -> str:
        """Load a system prompt from any path.

        Useful for loading prompts from outside the agent's prompts directory,
        such as evaluation prompts.

        Args:
            path: Path to the prompt file

        Returns:
            Prompt content as string

        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt not found: {path}")

        return path.read_text()
