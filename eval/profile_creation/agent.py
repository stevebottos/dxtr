"""
Evaluation Agent for Profile Creation Pipeline

Verifies the quality of:
1. Generated github summaries against source code
2. Synthesized profile against github analysis
"""

import json
from pathlib import Path

import sglang as sgl

from dxtr.agents.base import BaseAgent
from dxtr.config_v2 import config

from . import util


class Agent(BaseAgent):
    """Agent for evaluating profile creation pipeline outputs."""

    def __init__(self):
        super().__init__()
        self.temperature = 0.0  # Deterministic for evaluation
        self.max_tokens = config.agents.profile_manager_max_tokens
        self.prompts_dir = Path(__file__).parent

    @staticmethod
    @sgl.function
    def verify_summary_func(
        s, source_code, generated_analysis, system_prompt, max_tokens, temp, json_schema
    ):
        """SGLang function for verifying generated summaries."""
        s += sgl.system(system_prompt)
        s += sgl.user(
            f"Source code:\n```python\n{source_code}\n```\n\n"
            f"Generated analysis:\n{generated_analysis}"
        )
        s += sgl.assistant(
            sgl.gen(
                "verification",
                max_tokens=max_tokens,
                temperature=temp,
                json_schema=json_schema,
            )
        )

    @staticmethod
    @sgl.function
    def verify_profile_func(
        s, original_profile, github_summary, synthesized_profile, system_prompt, max_tokens, temp, json_schema
    ):
        """SGLang function for verifying synthesized profile."""
        s += sgl.system(system_prompt)
        s += sgl.user(
            f"Original Profile:\n{original_profile}\n\n"
            f"GitHub Analysis:\n{github_summary}\n\n"
            f"Synthesized Profile:\n{synthesized_profile}"
        )
        s += sgl.assistant(
            sgl.gen(
                "verification",
                max_tokens=max_tokens,
                temperature=temp,
                json_schema=json_schema,
            )
        )

    def verify_github_summary(self, github_summary: dict[str, str]) -> dict:
        """Verify generated github summaries against source code.

        Args:
            github_summary: Dict mapping file paths to generated JSON analyses

        Returns:
            Dict with file_results and aggregate metrics
        """
        system_prompt = self.load_system_prompt(self.prompts_dir / "summary_prompt.md")
        schema_json = json.dumps(util.SUMMARY_VERIFICATION_SCHEMA)

        batch_data = []
        file_paths = []

        for file_path, generated_analysis in github_summary.items():
            try:
                source_code = Path(file_path).read_text(encoding="utf-8")
                batch_data.append(
                    {
                        "source_code": source_code,
                        "generated_analysis": generated_analysis,
                        "system_prompt": system_prompt,
                        "max_tokens": self.max_tokens,
                        "temp": self.temperature,
                        "json_schema": schema_json,
                    }
                )
                file_paths.append(file_path)
            except Exception:
                continue

        print(f"Verifying {len(batch_data)} file analyses...")
        states = self.verify_summary_func.run_batch(
            batch_data, num_threads=128, progress_bar=True
        )

        # Parse results
        file_results = {}
        verification_results = []

        for i, state in enumerate(states):
            try:
                verification = json.loads(state["verification"])
                file_results[file_paths[i]] = verification
                verification_results.append(verification)
            except json.JSONDecodeError:
                file_results[file_paths[i]] = {"error": "Failed to parse verification"}
                verification_results.append({"error": "Failed to parse"})

        metrics = util.compute_summary_metrics(verification_results)

        return {"file_results": file_results, "metrics": metrics}

    def verify_profile(
        self,
        original_profile: str,
        github_summary: dict[str, str],
        synthesized_profile: str,
    ) -> dict:
        """Verify synthesized profile against source data.

        Args:
            original_profile: Original profile.md content
            github_summary: Dict mapping file paths to generated JSON analyses
            synthesized_profile: The synthesized profile content

        Returns:
            Dict with verification result and metrics
        """
        system_prompt = self.load_system_prompt(self.prompts_dir / "profile_prompt.md")
        schema_json = json.dumps(util.PROFILE_VERIFICATION_SCHEMA)

        # Format github summary for context
        github_summary_text = json.dumps(github_summary, indent=2)

        print("Verifying synthesized profile...")
        state = self.verify_profile_func.run(
            original_profile=original_profile,
            github_summary=github_summary_text,
            synthesized_profile=synthesized_profile,
            system_prompt=system_prompt,
            max_tokens=self.max_tokens,
            temp=self.temperature,
            json_schema=schema_json,
        )

        try:
            verification = json.loads(state["verification"])
            metrics = util.compute_profile_metrics(verification)
            return {"result": verification, "metrics": metrics}
        except json.JSONDecodeError:
            return {"result": {"error": "Failed to parse verification"}, "metrics": {"error": "Parse failed"}}

    def run(
        self,
        github_summary: dict[str, str],
        original_profile: str,
        synthesized_profile: str,
        output_dir: Path,
    ) -> dict:
        """Run full verification and save results.

        Args:
            github_summary: Dict mapping file paths to generated JSON analyses
            original_profile: Original profile.md content
            synthesized_profile: The synthesized profile content
            output_dir: Directory to save verification results

        Returns:
            Dict with summary_results, profile_results
        """
        # Verify github summaries
        summary_results = self.verify_github_summary(github_summary)

        # Verify synthesized profile
        profile_results = self.verify_profile(
            original_profile, github_summary, synthesized_profile
        )

        # Combine results
        results = {
            "summary_verification": summary_results,
            "profile_verification": profile_results,
        }

        # Save results
        verification_file = output_dir / "verification_results.json"
        verification_file.write_text(json.dumps(results, indent=2))

        # Print reports
        util.print_summary_metrics(
            summary_results["metrics"], summary_results["file_results"]
        )
        util.print_profile_metrics(profile_results["metrics"])

        print(f"\nFull results saved: {verification_file}")

        return results
