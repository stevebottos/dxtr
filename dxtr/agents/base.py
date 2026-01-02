"""
Base Agent for SGLang-based agents.

Provides common functionality for prompt loading and SGLang backend management.
"""

from pathlib import Path
import sglang as sgl

from dxtr.config_v2 import config


class SGLangAgent:
    """Base class for SGLang-powered agents."""

    def __init__(self, prompts_dir: Path | None = None):
        """Initialize SGLang agent with backend connection.

        Args:
            prompts_dir: Directory containing agent's prompt files.
                        Defaults to 'prompts' subdirectory of agent module.
        """
        # Connect to SGLang backend
        raw_url = config.sglang.base_url
        base_url = raw_url.replace("/v1", "").rstrip("/")

        try:
            self.backend = sgl.RuntimeEndpoint(base_url)
            sgl.set_default_backend(self.backend)
        except Exception as e:
            print(f"Connection failed to SGLang at {base_url}")
            raise e

        self.prompts_dir = prompts_dir

    def load_prompt(self, prompt_name: str) -> str:
        """Load a prompt from the agent's prompts directory.

        Args:
            prompt_name: Name of prompt file (without .md extension)

        Returns:
            Prompt content as string

        Raises:
            ValueError: If prompts_dir not set
            FileNotFoundError: If prompt file doesn't exist
        """
        if self.prompts_dir is None:
            raise ValueError("prompts_dir not set for this agent")

        prompt_file = self.prompts_dir / f"{prompt_name}.md"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt not found: {prompt_file}")

        return prompt_file.read_text()

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
