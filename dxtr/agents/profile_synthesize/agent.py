"""
Profile Manager Agent

Analyzes GitHub repositories and creates enriched user profiles using SGLang.
Task-focused agent with no conversational elements.
"""

import re
from pathlib import Path
import sglang as sgl
import json

from dxtr.config_v2 import config
from dxtr.agents.base import BaseAgent
from . import util


class Agent(BaseAgent):
    """Agent for GitHub analysis and profile creation using SGLang."""

    def __init__(self):
        """Initialize profile manager agent with SGLang backend."""
        super().__init__()
        self.temperature = config.agents.profile_manager_temperature
        self.max_tokens = config.agents.profile_manager_max_tokens
        self.system_prompt = self.load_system_prompt(
            Path(__file__).parent / "system.md"
        )

    @staticmethod
    @sgl.function
    def create_profile_func(s, context, system_prompt, max_tokens, temp):
        """SGLang function for the final profile enrichment step."""
        s += sgl.system(system_prompt)
        s += sgl.user(context)
        s += sgl.assistant(
            sgl.gen("enriched_profile", max_tokens=max_tokens, temperature=temp)
        )

    def run(
        self,
        profile_path: Path,
        output_dir: Path | None = None,
        additional_context: str | None = None,
    ) -> str:
        """Create enriched profile from profile.md file.

        Args:
            profile_path: Path to the seed profile.md
            output_dir: Directory to save outputs (github_summary.json, etc.)
                       Defaults to .dxtr in current directory
        """
        if output_dir is None:
            output_dir = config.paths.dxtr_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        profile_content = profile_path.read_text()
        enrichment_context = f"# Original Profile\n\n{profile_content}\n\n"

        enrichment_context = (
            enrichment_context + f"# Github Summary\n\n{additional_context}\n\n"
        )
        # Native SGLang execution for the final step
        final_state = self.create_profile_func.run(
            context=enrichment_context,
            system_prompt=self.system_prompt,
            max_tokens=self.max_tokens,
            temp=self.temperature,
        )

        result = final_state["enriched_profile"].strip()

        print(result)

        summary_file = output_dir / "profile.md"
        summary_file.write_text(result)
        return result
