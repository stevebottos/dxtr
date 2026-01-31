import json
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from dxtr import (
    load_system_prompt,
    github_summarizer,
    get_model_settings,
    log_tool_usage,
    data_models,
)
from dxtr.agents.subagents.util import parallel_map

from .util import (
    is_profile_url,
    fetch_profile_html,
    extract_pinned_repos,
    clone_repo,
)

from dxtr import util, constants

agent = Agent(
    github_summarizer,
    system_prompt=load_system_prompt(Path(__file__).parent / "system.md"),
    deps_type=data_models.GithubSummarizerRequest,  # GitHub profile base URL
)


@agent.tool
@log_tool_usage
async def summarize_repos(ctx: RunContext[data_models.GithubSummarizerRequest]) -> str:
    """Download and analyze the user's github content, required for a complete github summary."""
    repo_urls = ctx.deps.repo_urls

    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        cloned = []
        for repo_url in repo_urls:

            result = await clone_repo(repo_url, tmp)
            if result["success"]:
                cloned.append(result["path"])

        if not cloned:
            return "No repositories could be cloned"

        print(f"{len(cloned)}/{len(repo_urls)} successful.")
        print(f"Summarizing {len(cloned)} repos...")

        all_files = []
        for repo_path in cloned:
            path = Path(repo_path)
            if not path.exists():
                continue

            python_files = list(path.rglob("*.py"))
            for py_file in python_files:
                if py_file.name == "__init__.py":
                    continue  # Skip __init__.py files entirely
                try:
                    content = py_file.read_text(encoding="utf-8")
                    if len(content.strip()) > 120:  # Skip tiny/empty files
                        all_files.append(
                            {
                                "repo_path": repo_path,
                                "path": str(py_file.relative_to(path)),
                                "content": content,
                            }
                        )
                except Exception as e:
                    print(f"Warning: Failed to read {py_file}: {e}")
                    continue

        if not all_files:
            return "No Python files found to analyze"

        async def summarize_one(file_info: dict, idx: int, total: int) -> dict:
            file_path = file_info["path"]
            try:
                result = await agent.run(
                    f"Analyze this file ({file_path}):\n\n```python\n{file_info['content']}\n```",
                    model_settings=get_model_settings(),
                )
                print(f"  ✓ [{idx}/{total}] {file_path}")
                return {
                    "repo_path": file_info["repo_path"],
                    "file": file_path,
                    "analysis": result.output,
                }
            except Exception as e:
                print(f"  ✗ [{idx}/{total}] {file_path} (ERROR: {e})")
                return {
                    "repo_path": file_info["repo_path"],
                    "file": file_path,
                    "error": str(e),
                }

        file_summaries = await parallel_map(
            all_files,
            summarize_one,
            desc="Analyzing files",
            status_interval=10.0,
        )

        # Group by repo
        repo_summaries = {}
        for summary in file_summaries:
            rp = summary.pop("repo_path")
            if rp not in repo_summaries:
                repo_summaries[rp] = []
            repo_summaries[rp].append(summary)

        all_summaries = [
            {"repo_path": rp, "files_analyzed": len(files), "file_summaries": files}
            for rp, files in repo_summaries.items()
        ]

        summary_file = tmp / "github_summary.json"
        summary_file.write_text(json.dumps(all_summaries, indent=2))
        await util.upload_to_gcs(
            str(summary_file),
            str(
                Path(constants.profiles_dir.format(user_id=ctx.deps.user_id))
                / "github_summary.json"
            ),
        )

    return "github_summary.json uploaded to the user's profile."
