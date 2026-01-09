"""
GitHub Summarize Agent

Analyzes GitHub repositories and produces structured summaries of code modules.
"""

import json
import re
from pathlib import Path

import sglang as sgl

from dxtr.agents.base import BaseAgent
from dxtr.config_v2 import config

from . import utils


class Agent(BaseAgent):
    """Agent for analyzing GitHub repositories using SGLang."""

    def __init__(self):
        super().__init__()
        self.temperature = config.agents.profile_manager_temperature
        self.max_tokens = config.agents.profile_manager_max_tokens
        self.system_prompt = self.load_system_prompt(
            Path(__file__).parent / "system.md"
        )

    def _extract_github_url(self, profile_content: str) -> str | None:
        """Extract GitHub profile URL from profile.md content."""
        url_pattern = r'https?://github\.com/[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, profile_content)
        for url in urls:
            if utils.is_profile_url(url):
                return url
        return None

    @staticmethod
    @sgl.function
    def summarize_file_func(
        s, source_code, system_prompt, max_tokens, temp, json_schema
    ):
        """Parallelizable SGLang function for analyzing code modules."""
        s += sgl.system(system_prompt)
        s += sgl.user(f"Analyze this source code:\n\n{source_code}")
        s += sgl.assistant(
            sgl.gen(
                "analysis",
                max_tokens=max_tokens,
                temperature=temp,
                frequency_penalty=1.1,
                presence_penalty=0.1,
                json_schema=json_schema,
            )
        )

    def analyze_github_repos(self, github_url: str) -> dict[str, str]:
        """Analyze repositories using SGLang's native parallel batching."""
        if not utils.is_profile_url(github_url):
            return {}

        html = utils.fetch_profile_html(github_url)
        if not html:
            return {}

        pinned_repos = utils.extract_pinned_repos(html)
        pinned_repos = [repo for repo in pinned_repos if not repo.endswith("/dxtr-cli")]

        successful = []
        for repo_url in pinned_repos:
            result = utils.clone_repo(repo_url)
            if result["success"]:
                successful.append(result)

        github_summary = {}

        python_files = []
        for result in successful:
            repo_path = Path(result["path"])
            python_files.extend(utils.find_python_files(repo_path))

        batch_data = []
        file_paths = []
        schema_json = json.dumps(utils.MODULE_ANALYSIS_SCHEMA)

        for py_file in python_files:
            try:
                batch_data.append(
                    {
                        "source_code": py_file.read_text(encoding="utf-8"),
                        "system_prompt": self.system_prompt,
                        "max_tokens": self.max_tokens,
                        "temp": self.temperature,
                        "json_schema": schema_json,
                    }
                )
                file_paths.append(str(py_file))
            except Exception:
                continue

        states = self.summarize_file_func.run_batch(
            batch_data, num_threads=128, progress_bar=True
        )

        for i, state in enumerate(states):
            github_summary[file_paths[i]] = state["analysis"]

        return github_summary

    def run(self, profile_path: Path, output_dir: Path | None = None) -> dict:
        """Run the github summarization agent.

        Args:
            profile_path: Path to the profile.md file containing GitHub URL
            output_dir: Directory to save github_summary.json

        Returns:
            Dict mapping file paths to their analysis JSON strings
        """
        if output_dir is None:
            output_dir = config.paths.dxtr_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        profile_content = profile_path.read_text()
        github_url = self._extract_github_url(profile_content)

        if github_url:
            github_summary = self.analyze_github_repos(github_url)
            if github_summary:
                summary_file = output_dir / "github_summary.json"
                summary_file.write_text(json.dumps(github_summary, indent=2))
            return github_summary

        return {}
