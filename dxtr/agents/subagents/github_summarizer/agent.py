import json
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from dxtr import DXTR_DIR, load_system_prompt, github_summarizer, get_model_settings, log_tool_usage
from dxtr.agents.subagents.util import parallel_map

from .util import (
    is_profile_url,
    fetch_profile_html,
    extract_pinned_repos,
    clone_repo,
    find_python_files,
)


agent = Agent(
    github_summarizer,
    system_prompt=load_system_prompt(Path(__file__).parent / "system.md"),
    deps_type=str,  # GitHub profile base URL
)


class CloneReposRequest(BaseModel):
    repo_urls: list[str] = Field(
        description="List of GitHub repository URLs to clone.",
        examples=[["https://github.com/user/repo1", "https://github.com/user/repo2"]],
    )


class SummarizeReposRequest(BaseModel):
    repo_paths: list[str] = Field(
        description="List of local filesystem paths to cloned repositories.",
        examples=[["/home/user/.dxtr/repos/owner/repo"]],
    )


@agent.tool
@log_tool_usage
async def get_pinned_repos(ctx: RunContext[str]) -> list[str]:
    """Fetch pinned repository URLs from the user's GitHub profile.
    If you have already called this function in your history, don't worry about calling it again."""
    github_url = ctx.deps

    if not is_profile_url(github_url):
        return [f"Error: Not a valid GitHub profile URL: {github_url}"]

    html = fetch_profile_html(github_url)
    if not html:
        return ["Error: Could not fetch GitHub profile page"]

    pinned_repos = extract_pinned_repos(html)
    if not pinned_repos:
        return ["No pinned repositories found on profile"]

    return pinned_repos


@agent.tool_plain
@log_tool_usage
async def clone_repos(request: CloneReposRequest) -> str:
    """Clone GitHub repositories to local disk."""
    print(f"Cloning {len(request.repo_urls)} repos...")

    cloned = []
    for repo_url in request.repo_urls:
        result = clone_repo(repo_url, DXTR_DIR)
        if result["success"]:
            cloned.append(
                {
                    "path": result["path"],
                    "name": f"{result.get('owner', 'unknown')}/{result.get('repo', 'unknown')}",
                    "url": result["url"],
                }
            )

    if not cloned:
        return "No repositories could be cloned"

    return json.dumps(
        {
            "repos_cloned": len(cloned),
            "repos": cloned,
        },
        indent=2,
    )


@agent.tool_plain
@log_tool_usage
async def summarize_repos(request: SummarizeReposRequest) -> str:
    """Analyze Python files across cloned repositories. Saves results to ~/.dxtr/github_summary.json."""
    print(f"Summarizing {len(request.repo_paths)} repos...")

    all_files = []
    for repo_path in request.repo_paths:
        path = Path(repo_path)
        if not path.exists():
            continue

        python_files = find_python_files(path)
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
            except Exception:
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

    # Save to github_summary.json
    result = {"repos_analyzed": len(request.repo_paths), "summaries": all_summaries}

    summary_file = DXTR_DIR / "github_summary.json"
    summary_file.write_text(json.dumps(result, indent=2))

    return f"GitHub analysis complete. Saved to {summary_file}.\n\n{json.dumps(result, indent=2)}"
