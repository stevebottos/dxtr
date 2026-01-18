from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from dxtr import DXTR_DIR, master, load_system_prompt, get_model_settings, run_agent
from dxtr.agents.subagents import github_summarizer
from dxtr.agents.subagents import profile_synthesis
from dxtr.agents.subagents import papers_ranking
from dxtr.agents.subagents.papers_ranking import util as papers_ranking_util
from dxtr.agents.util import (
    get_available_dates,
    download_papers,
    load_papers_metadata,
    format_available_dates,
)


SYSTEM_PROMPT = load_system_prompt(Path(__file__).parent / "system.md")

agent = Agent(master, system_prompt=SYSTEM_PROMPT)


class GitHubProfileRequest(BaseModel):
    base_url: str = Field(
        description="GitHub profile BASE URL only, format: https://github.com/<username>. "
        "Must NOT be a repository URL (e.g. https://github.com/user/repo).",
        examples=["https://github.com/stevebottos", "https://github.com/anthropics"],
    )


class FileReadRequest(BaseModel):
    file_path: str = Field(
        description="Absolute or relative path to a file to read.",
        examples=["~/.profile.md", "/home/user/documents/resume.txt"],
    )


class ProfileSynthesisRequest(BaseModel):
    seed_profile: str = Field(
        description="The user's self-description content (from their profile file).",
    )
    github_summary: str = Field(
        description="The JSON summary from GitHub analysis.",
    )


@agent.tool_plain
async def call_github_summarizer(request: GitHubProfileRequest) -> str:
    """Analyze a user's GitHub profile: clone pinned repos, read code, generate summary."""
    print("Call: Github Summary Agent")
    result = await run_agent(
        github_summarizer.agent,
        "Analyze the user's GitHub profile.",
        deps=request.base_url,
        model_settings=get_model_settings(),
    )
    return result.output


@agent.tool_plain
async def read_file(request: FileReadRequest) -> str:
    """Read content from a file."""
    try:
        path = Path(request.file_path).expanduser().resolve()
        if not path.exists():
            return f"Error: File not found: {request.file_path}"
        return path.read_text()
    except Exception as e:
        return f"Error reading file: {e}"


@agent.tool_plain
async def call_profile_synthesizer(request: ProfileSynthesisRequest) -> str:
    """Synthesize an enriched user profile from seed profile and GitHub analysis.
    If the user has provided a github profile, you need to handle that first."""
    print("Call: Profile Synthesis Agent")
    deps = profile_synthesis.ProfileSynthesisDeps(
        seed_profile=request.seed_profile,
        github_summary=request.github_summary,
    )
    result = await run_agent(
        profile_synthesis.agent,
        f"Create an enriched profile.\n\nSeed Profile:\n{request.seed_profile}\n\nGitHub Analysis:\n{request.github_summary}",
        deps=deps,
        model_settings=get_model_settings(),
    )

    # Save synthesized profile
    profile_file = DXTR_DIR / "synthesized_profile.md"
    profile_file.write_text(result.output)
    print(f"  Saved to {profile_file}")

    return result.output


# === State Tools ===


@agent.tool_plain
async def get_today() -> str:
    """Get today's date in YYYY-MM-DD format."""
    from datetime import datetime
    return datetime.today().strftime("%Y-%m-%d")


@agent.tool_plain
async def check_profile_state() -> str:
    """Check the current state of the user's DXTR profile directory.

    Returns what artifacts exist in ~/.dxtr including:
    - synthesized_profile.md (enriched user profile)
    - github_summary.json (GitHub analysis results)
    - papers/ directory with downloaded papers

    Use this to determine what work needs to be done (e.g., profile synthesis).
    """
    print("Call: check_profile_state")

    lines = [f"DXTR directory: {DXTR_DIR}", ""]

    # Check synthesized profile
    profile_path = DXTR_DIR / "synthesized_profile.md"
    if profile_path.exists():
        size = profile_path.stat().st_size
        lines.append(f"[x] synthesized_profile.md ({size} bytes)")
    else:
        lines.append("[ ] synthesized_profile.md (not created)")

    # Check GitHub summary
    github_path = DXTR_DIR / "github_summary.json"
    if github_path.exists():
        size = github_path.stat().st_size
        lines.append(f"[x] github_summary.json ({size} bytes)")
    else:
        lines.append("[ ] github_summary.json (not created)")

    # Check cloned repos
    repos_dir = DXTR_DIR / "repos"
    if repos_dir.exists():
        repo_count = sum(1 for p in repos_dir.iterdir() if p.is_dir())
        lines.append(f"[x] repos/ ({repo_count} repositories cloned)")
    else:
        lines.append("[ ] repos/ (no repositories cloned)")

    # Check papers
    papers_dir = DXTR_DIR / "papers"
    if papers_dir.exists():
        dates = [d.name for d in papers_dir.iterdir() if d.is_dir()]
        if dates:
            total_papers = sum(
                1 for d in papers_dir.iterdir() if d.is_dir()
                for p in d.iterdir() if p.is_dir() and (p / "metadata.json").exists()
            )
            lines.append(f"[x] papers/ ({len(dates)} dates, {total_papers} papers)")
        else:
            lines.append("[ ] papers/ (empty)")
    else:
        lines.append("[ ] papers/ (not created)")

    return "\n".join(lines)


# === Paper Tools ===


class GetPapersRequest(BaseModel):
    days_back: int = Field(
        default=7,
        description="Number of days to look back for available papers.",
    )


class DownloadPapersRequest(BaseModel):
    dates: list[str] = Field(
        description="List of dates to download papers for (format: YYYY-MM-DD).",
        examples=[["2024-01-15"], ["2024-01-14", "2024-01-15"]],
    )
    paper_ids: list[str] | None = Field(
        default=None,
        description="Optional list of specific paper IDs to download. If None, downloads all papers for the dates.",
    )


class RankPapersRequest(BaseModel):
    date: str = Field(
        description="Date to rank papers for (format: YYYY-MM-DD).",
        examples=["2024-01-15"],
    )


@agent.tool_plain
async def get_papers(request: GetPapersRequest) -> str:
    """Check available papers from the past week.

    Returns a summary of dates and paper counts. Use this to see what papers
    are already downloaded before asking the user to select dates.
    """
    print("Call: get_papers")
    available = get_available_dates(days_back=request.days_back)
    return format_available_dates(available)


@agent.tool_plain
async def fetch_and_download_papers(request: DownloadPapersRequest) -> str:
    """Download papers for specified dates from HuggingFace/ArXiv.

    Downloads paper metadata (and optionally PDFs) for the given dates.
    Use this after confirming with the user which dates to download.
    """
    print(f"Call: fetch_and_download_papers for {request.dates}")

    total_downloaded = 0
    results = []

    for date in request.dates:
        downloaded = download_papers(
            date=date,
            paper_ids=request.paper_ids,
            download_pdfs=False,  # Per CLAUDE.md - Gemini handles PDFs directly
        )
        count = len(downloaded)
        total_downloaded += count
        results.append(f"{date}: {count} papers")

    return (
        f"Downloaded papers for {len(request.dates)} dates:\n"
        + "\n".join(f"  - {r}" for r in results)
        + f"\n\nTotal: {total_downloaded} papers"
    )


@agent.tool_plain
async def rank_papers(request: RankPapersRequest) -> str:
    """Rank papers for a date against the user's synthesized profile."""
    print(f"Call: rank_papers for {request.date}")

    # Load user profile
    profile = papers_ranking_util.load_profile()
    if "No synthesized profile found" in profile:
        return profile

    # Load papers and convert to dict
    papers_list = load_papers_metadata(request.date)
    if not papers_list:
        return f"No papers found for {request.date}. Use fetch_and_download_papers first."

    papers_dict = papers_ranking_util.papers_list_to_dict(papers_list)

    # Rank papers in parallel
    results = await papers_ranking.rank_papers_parallel(profile, papers_dict)

    # Format results
    rankings_text = papers_ranking_util.format_ranking_results(results)

    return f"Ranked {len(results)} papers\n\n{rankings_text}"
