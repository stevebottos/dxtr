"""
Profile Manager Agent

Analyzes GitHub repositories and creates enriched user profiles using SGLang.
Task-focused agent with no conversational elements.
"""

import json
import re
from pathlib import Path

import sglang as sgl

from dxtr.agents.base import BaseAgent
from dxtr.config_v2 import config

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

    def _extract_github_url(self, profile_content: str) -> str | None:
        """Extract GitHub profile URL from profile.md content."""
        url_pattern = r'https?://github\.com/[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, profile_content)
        for url in urls:
            if util.is_profile_url(url):
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
                frequency_penalty=1.1,  # Values between 1.0 and 1.5 usually fix loops
                presence_penalty=0.1,
                json_schema=json_schema,
            )
        )

    # @staticmethod
    # @sgl.function
    # def create_profile_func(s, context, system_prompt, max_tokens, temp):
    #     """SGLang function for the final profile enrichment step."""
    #     s += sgl.system(system_prompt)
    #     s += sgl.user(context)
    #     s += sgl.assistant(
    #         sgl.gen("enriched_profile", max_tokens=max_tokens, temperature=temp)
    #     )

    # @staticmethod
    # @sgl.function
    # def verify_summary_func(
    #     s, source_code, generated_analysis, system_prompt, max_tokens, temp, json_schema
    # ):
    #     """Parallelizable SGLang function for verifying generated summaries."""
    #     s += sgl.system(system_prompt)
    #     s += sgl.user(
    #         f"Source code:\n```python\n{source_code}\n```\n\n"
    #         f"Generated analysis:\n{generated_analysis}"
    #     )
    #     s += sgl.assistant(
    #         sgl.gen(
    #             "verification",
    #             max_tokens=max_tokens,
    #             temperature=temp,
    #             json_schema=json_schema,
    #         )
    #     )

    def analyze_github_repos(self, github_url: str) -> dict[str, str]:
        """Analyze repositories using SGLang's native parallel batching."""
        if not util.is_profile_url(github_url):
            return {}

        html = util.fetch_profile_html(github_url)
        if not html:
            return {}

        pinned_repos = util.extract_pinned_repos(html)
        pinned_repos = [repo for repo in pinned_repos if not repo.endswith("/dxtr-cli")]

        successful = []
        for repo_url in pinned_repos:
            result = util.clone_repo(repo_url)
            if result["success"]:
                successful.append(result)

        github_summary = {}

        python_files = []
        for result in successful:
            repo_path = Path(result["path"])
            python_files.extend(util.find_python_files(repo_path))

        batch_data = []
        file_paths = []
        # Convert schema to JSON string for SGLang
        schema_json = json.dumps(util.MODULE_ANALYSIS_SCHEMA)

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

        # This utilizes SGLang's Prefix Caching for the system_prompt
        states = self.summarize_file_func.run_batch(
            batch_data, num_threads=128, progress_bar=True
        )

        for i, state in enumerate(states):
            github_summary[file_paths[i]] = state["analysis"]

        return github_summary

    def run(self, profile_path: Path, output_dir: Path | None = None) -> dict:
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

        #         repos = set()
        #         for file_path in github_summary.keys():
        #             parts = Path(file_path).parts
        #             if "repos" in parts:
        #                 idx = parts.index("repos")
        #                 if idx + 2 < len(parts):
        #                     repos.add(f"{parts[idx + 1]}/{parts[idx + 2]}")
        #
        #         enrichment_context += "# GitHub Analysis\n\n"
        #         enrichment_context += f"Analyzed {len(github_summary)} files across {len(repos)} repositories.\n\n"
        #         enrichment_context += f"Repositories: {', '.join(sorted(repos))}\n\n"
        #
        #         # Append the specific file analyses to context
        #         for path, analysis in github_summary.items():
        #             enrichment_context += f"### File: {path}\n{analysis}\n\n"
        #     else:
        #         enrichment_context += (
        #             "# GitHub Analysis\n\nNo repositories analyzed.\n\n"
        #         )
        # else:
        #     enrichment_context += "# GitHub Analysis\n\nNo GitHub URL found.\n\n"
        #
        # # Native SGLang execution for the final step
        # final_state = self.create_profile_func.run(
        #     context=enrichment_context,
        #     system_prompt=self.load_prompt("profile_creation"),
        #     max_tokens=self.max_tokens,
        #     temp=self.temperature,
        # )
        #
        # result = final_state["enriched_profile"].strip()
        # return result
        #

    # def verify_github_summary(
    #     self, github_summary: dict[str, str], prompt_path: Path | str
    # ) -> dict:
    #     """Verify generated github summaries against source code.
    #
    #     Args:
    #         github_summary: Dict mapping file paths to generated JSON analyses
    #         prompt_path: Path to the verification prompt file
    #
    #     Returns:
    #         Dict with verification results and aggregate metrics
    #     """
    #     system_prompt = self.load_system_prompt(prompt_path)
    #     schema_json = json.dumps(analyzer.VERIFICATION_SCHEMA)
    #
    #     batch_data = []
    #     file_paths = []
    #
    #     for file_path, generated_analysis in github_summary.items():
    #         try:
    #             source_code = Path(file_path).read_text(encoding="utf-8")
    #             batch_data.append(
    #                 {
    #                     "source_code": source_code,
    #                     "generated_analysis": generated_analysis,
    #                     "system_prompt": system_prompt,
    #                     "max_tokens": self.max_tokens,
    #                     "temp": 0.0,  # Deterministic for evaluation
    #                     "json_schema": schema_json,
    #                 }
    #             )
    #             file_paths.append(file_path)
    #         except Exception:
    #             continue
    #
    #     states = self.verify_summary_func.run_batch(
    #         batch_data, num_threads=128, progress_bar=True
    #     )
    #
    #     # Parse results and compute metrics
    #     results = {}
    #     keyword_scores = []
    #     summary_scores = []
    #
    #     for i, state in enumerate(states):
    #         try:
    #             verification = json.loads(state["verification"])
    #             results[file_paths[i]] = verification
    #             keyword_scores.append(verification.get("keywords_score", 0))
    #             summary_scores.append(verification.get("summary_score", 0))
    #         except json.JSONDecodeError:
    #             results[file_paths[i]] = {"error": "Failed to parse verification"}
    #
    #     # Aggregate metrics
    #     metrics = {
    #         "total_files": len(file_paths),
    #         "avg_keywords_score": sum(keyword_scores) / len(keyword_scores)
    #         if keyword_scores
    #         else 0,
    #         "avg_summary_score": sum(summary_scores) / len(summary_scores)
    #         if summary_scores
    #         else 0,
    #         "keywords_score_distribution": {
    #             score: keyword_scores.count(score) for score in range(1, 6)
    #         },
    #         "summary_score_distribution": {
    #             score: summary_scores.count(score) for score in range(1, 6)
    #         },
    #     }
    #
    #     return {"file_results": results, "metrics": metrics}
