"""
Git Helper Agent

INTERNAL HELPER - NOT A USER-FACING TOOL
This agent is called programmatically by the profile_creator agent.
It has no TOOL_DEFINITION and is not exposed to the main chat agent.

Handles GitHub repository analysis:
1. Scrapes pinned repos from GitHub profiles
2. Clones repositories
3. Analyzes Python code using LLM
4. Returns github_summary.json data
"""

import json
from pathlib import Path
from ollama import chat
from dxtr.base import Agent
from dxtr.config import config
from .tools import git_tools
from . import analyzer


class GitHelperAgent(Agent):
    """Agent for analyzing GitHub repositories and Python code."""

    def __init__(self):
        """Initialize git helper agent."""
        model_config = config.get_model_config("git_helper")
        super().__init__(
            name="git_helper",
            model=model_config.name,
            prompts_dir=Path(__file__).parent / "prompts",
            default_options={
                "temperature": model_config.temperature,
                "num_ctx": model_config.context_window,
            }
        )

    def run(self, github_url: str) -> dict[str, str]:
        """
        Analyze GitHub pinned repositories.

        This is the main entry point for the git_helper agent.

        Args:
            github_url: GitHub profile URL

        Returns:
            Dict mapping file paths to JSON analysis strings
            Format: {"/path/to/file.py": '{"keywords": [...], "summary": "..."}', ...}
        """
        print(f"\n[Git Helper Agent: Analyzing {github_url}]")

        if not git_tools.is_profile_url(github_url):
            print(f"  [Error: Not a GitHub profile URL]")
            return {}

        html = git_tools.fetch_profile_html(github_url)
        if not html:
            print("  [Failed to fetch profile HTML]")
            return {}

        pinned_repos = git_tools.extract_pinned_repos(html)
        pinned_repos = [repo for repo in pinned_repos if not repo.endswith('/dxtr-cli')]

        if not pinned_repos:
            print("  [No pinned repositories found]")
            return {}

        print(f"  [Found {len(pinned_repos)} pinned repository(ies)]")

        print(f"\n[Cloning repositories...]")
        clone_results = []
        for repo_url in pinned_repos:
            result = git_tools.clone_repo(repo_url)
            clone_results.append(result)

            if result["success"]:
                status = "✓ cached" if "cached" in result["message"].lower() else "✓ cloned"
                print(f"  [{status}] {result['owner']}/{result['repo']}")
            else:
                print(f"  [✗ failed] {result['url']}: {result['message']}")

        successful = [r for r in clone_results if r["success"]]
        if not successful:
            print("  [No repositories to analyze]")
            return {}

        print(f"\n[Analyzing {len(successful)} repository(ies)...]")

        github_summary = {}

        for result in successful:
            repo_path = Path(result["path"])
            print(f"  [Analyzing {result['owner']}/{result['repo']}...]")

            python_files = analyzer.find_python_files(repo_path)

            if not python_files:
                print(f"    [No Python files found]")
                continue

            print(f"    [Found {len(python_files)} Python file(s)]")

            for idx, py_file in enumerate(python_files, 1):
                rel_path = py_file.relative_to(repo_path)
                print(f"    [{idx}/{len(python_files)}] {rel_path}", end=" ", flush=True)

                try:
                    source_code = py_file.read_text(encoding='utf-8')
                    file_size_kb = len(source_code) / 1024

                    # Use chat with format parameter for structured output
                    response = chat(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.prompts.load("module_analysis")},
                            {"role": "user", "content": source_code}
                        ],
                        options=self.default_options,
                        format=analyzer.MODULE_ANALYSIS_SCHEMA
                    )

                    github_summary[str(py_file)] = response.message.content

                    prompt_tokens = response.get("prompt_eval_count", 0)
                    completion_tokens = response.get("eval_count", 0)
                    print(f"({file_size_kb:.1f}KB, {prompt_tokens + completion_tokens} tokens)")

                except Exception as e:
                    print(f"[ERROR: {str(e)}]")
                    continue

        print(f"\n  [✓] Git Helper Agent: Analyzed {len(github_summary)} file(s) total")

        return github_summary


# Global instance for backward compatibility
_agent = GitHelperAgent()


def run(github_url: str) -> dict[str, str]:
    """
    Analyze GitHub pinned repositories.

    This is a convenience function that delegates to the agent instance.

    Args:
        github_url: GitHub profile URL

    Returns:
        Dict mapping file paths to JSON analysis strings
    """
    return _agent.run(github_url)
