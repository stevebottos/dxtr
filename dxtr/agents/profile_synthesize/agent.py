"""
Profile Synthesize Agent

Synthesizes a comprehensive user profile from:
1. The seed profile.md provided by user
2. Artifacts in .dxtr/ directory (github_summary.json, resume_summary.json, etc.)

Outputs to .dxtr/profile.md
"""

import json
import re
from pathlib import Path

import sglang as sgl

from dxtr.config_v2 import config
from dxtr.agents.base import BaseAgent


class Agent(BaseAgent):
    """Agent for synthesizing enriched profiles from available artifacts."""

    def __init__(self):
        """Initialize profile synthesize agent."""
        super().__init__()
        self.temperature = config.agents.profile_manager_temperature
        self.max_tokens = config.agents.profile_manager_max_tokens
        self.system_prompt = self.load_system_prompt(
            Path(__file__).parent / "system.md"
        )

    @staticmethod
    @sgl.function
    def create_profile_func(s, context, system_prompt, max_tokens, temp):
        """SGLang function for profile synthesis."""
        s += sgl.system(system_prompt)
        s += sgl.user(context)
        s += sgl.assistant(
            sgl.gen("enriched_profile", max_tokens=max_tokens, temperature=temp)
        )

    def _read_artifact(self, artifact_path: Path) -> str | None:
        """Read an artifact file if it exists."""
        if artifact_path.exists():
            try:
                content = artifact_path.read_text()
                # For JSON files, pretty-print for readability
                if artifact_path.suffix == ".json":
                    data = json.loads(content)
                    return json.dumps(data, indent=2)
                return content
            except Exception:
                return None
        return None

    def _gather_artifacts(self) -> dict[str, str]:
        """
        Gather all available artifacts from .dxtr/ directory.

        Returns:
            Dict mapping artifact name to content
        """
        artifacts = {}
        dxtr_dir = config.paths.dxtr_dir

        # Define known artifact files
        artifact_files = {
            "github_summary": dxtr_dir / "github_summary.json",
            "resume_summary": dxtr_dir / "resume_summary.json",
            "website_summary": dxtr_dir / "website_summary.json",
        }

        for name, path in artifact_files.items():
            content = self._read_artifact(path)
            if content:
                artifacts[name] = content

        return artifacts

    def run(self, seed_profile_path: Path) -> str:
        """
        Create enriched profile from seed profile and .dxtr/ artifacts.

        Args:
            seed_profile_path: Path to the user's seed profile.md

        Returns:
            The synthesized profile content
        """
        output_dir = config.paths.dxtr_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Read seed profile
        seed_profile_path = Path(seed_profile_path)
        if not seed_profile_path.exists():
            raise FileNotFoundError(f"Seed profile not found: {seed_profile_path}")

        seed_content = seed_profile_path.read_text()

        # Gather all available artifacts
        artifacts = self._gather_artifacts()

        # Build context for synthesis
        context_parts = [f"# Seed Profile\n\n{seed_content}"]

        for name, content in artifacts.items():
            # Convert artifact name to readable title
            title = name.replace("_", " ").title()
            context_parts.append(f"# {title}\n\n{content}")

        enrichment_context = "\n\n---\n\n".join(context_parts)

        # Run synthesis
        final_state = self.create_profile_func.run(
            context=enrichment_context,
            system_prompt=self.system_prompt,
            max_tokens=self.max_tokens,
            temp=self.temperature,
        )

        result = final_state["enriched_profile"].strip()

        # Clean think tags if present
        clean_text = re.sub(r"<think>[\s\S]*?</think>", "", result).strip()

        # Save to configured profile path
        output_file = config.paths.profile_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(clean_text)

        return clean_text
