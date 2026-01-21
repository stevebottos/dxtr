from pathlib import Path
from tempfile import TemporaryDirectory
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from dxtr import master, load_system_prompt, get_model_settings, run_agent, log_tool_usage
from dxtr.agents.subagents import github_summarizer
from dxtr.agents.subagents import profile_synthesis
from dxtr.agents.subagents import papers_ranking
from dxtr.agents.util import (
    get_available_dates,
    fetch_papers_for_date,
    download_papers as do_download_papers,
    load_papers_metadata,
    format_available_dates,
    papers_list_to_dict,
    format_ranking_results,
)

from dxtr import util, constants, data_models

SYSTEM_PROMPT = load_system_prompt(Path(__file__).parent / "system.md")

agent = Agent(master, system_prompt=SYSTEM_PROMPT, deps_type=data_models.MasterRequest)


@agent.tool
@log_tool_usage
async def check_profile_state(ctx: RunContext[data_models.MasterRequest]) -> str:
    """Check the current state of the user's profile.

    Returns what artifacts exist in the user's profile, including:
    - synthesized_profile.md (enriched user profile)
    - github_summary.json (GitHub analysis results)
    - papers/ directory with downloaded papers

    Use this to determine what work needs to be done (e.g., profile synthesis, github summary, etc.),
    as it's quite possible that you have already produced the necessary artifacts.
    """
    profile_folder = constants.profiles_dir.format(user_id=ctx.deps.user_id)
    result = await util.listdir_gcs(profile_folder)

    if not len(result):
        return "There are no database files for this user. Create their profile from scratch."
    return f"Current database files for this user: {result}"


@agent.tool
@log_tool_usage
async def get_github_summary(ctx: RunContext[data_models.MasterRequest]) -> str:
    """Retrieves the user's github summary from our database. If it's available, retrieve
    the profile before producing the profile summary. If it's not available, recompute it."""
    profile_path = Path(constants.profiles_dir.format(user_id=ctx.deps.user_id))
    content = await util.read_json_from_gcs(str(profile_path / "github_summary.json"))

    if not len(content):
        print("NO GITHUB SUMMARY")
        return "No github summary exists for the user in our database."

    return f"The user's github summary is as follows:\n{content}"


@agent.tool
@log_tool_usage
async def get_user_profile(ctx: RunContext[data_models.MasterRequest]) -> str:
    """Retrieves the user's synthesized profile from our database. If it is not listed as available by
    check_profile_state, then we need to create it."""
    profile_path = Path(constants.profiles_dir.format(user_id=ctx.deps.user_id))
    content = await util.read_from_gcs(str(profile_path / "synthesized_profile.md"))

    if not len(content):
        print("NO USER DATA")
        return "No github summary exists for the user in our database."

    return f"The user's profile is as follows:\n{content}"


@agent.tool
@log_tool_usage
async def create_github_summary(
    ctx: RunContext[data_models.MasterRequest], request: data_models.GithubSummarizerRequest
) -> str:
    """Analyze the repo or repos that a user has provided in order to summarize their skills/knowledge
    based on the code. Should be called before attempting to create a profile.
    """
    if not len(request.repo_urls):
        return "The user has not provided any repos to summarize."

    dep = data_models.GithubSummarizerRequest(repo_urls=request.repo_urls, **ctx.deps.model_dump())
    result = await run_agent(
        github_summarizer.agent,
        "Analyze the user's GitHub profile.",
        deps=dep,
        model_settings=get_model_settings(),
    )
    return result.output


from typing import List


def extract_chat_only(messages: List) -> str:
    chat_lines = []

    for msg in messages:
        for part in getattr(msg, "parts", []):
            # Only include user content or tool outputs
            if part.__class__.__name__ in ("UserPromptPart", "ToolReturnPart"):
                chat_lines.append(part.content)

    return "\n".join(chat_lines)


@agent.tool
@log_tool_usage
async def call_profile_synthesizer(ctx: RunContext[data_models.MasterRequest]) -> str:
    """Synthesize an enriched user profile from seed profile and GitHub analysis.
    If the user has provided a github profile, you need to handle that first.
    Make sure you check the user's profile to see if they have a github summary first.
    Make sure that the user provides you with explicit information about them as well, you cannot
    produce a sufficient result with just a github. If you need some more information,
    you may ask the user 3-5 questions.
    """

    chat_history = extract_chat_only(ctx.messages)
    result = await run_agent(
        profile_synthesis.agent,
        f"Create an enriched profile using the following chat: {chat_history}.",
        deps=None,
        model_settings=get_model_settings(),
    )

    # Save synthesized profile
    with TemporaryDirectory() as tmp:
        profile_file = Path(tmp) / "synthesized_profile.md"
        profile_file.write_text(result.output)
        await util.upload_to_gcs(
            str(profile_file),
            str(
                Path(constants.profiles_dir.format(user_id=ctx.deps.user_id))
                / "synthesized_profile.md"
            ),
        )
    print(f"  Saved to {profile_file}")

    return result.output


# === State Tools ===


@agent.tool_plain
@log_tool_usage
async def get_today() -> str:
    """Get today's date in YYYY-MM-DD format."""
    from datetime import datetime

    return datetime.today().strftime("%Y-%m-%d")


# === Paper Tools ===


class GetPapersRequest(BaseModel):
    days_back: int = Field(
        default=7,
        description="Number of days to look back for available papers.",
    )


class FetchPapersRequest(BaseModel):
    date: str = Field(
        description="Date to fetch papers for (format: YYYY-MM-DD).",
        examples=["2024-01-15"],
    )


class DownloadPapersRequest(BaseModel):
    date: str = Field(
        description="Date to download papers for (format: YYYY-MM-DD).",
        examples=["2024-01-15"],
    )
    paper_ids: list[str] | None = Field(
        default=None,
        description="Optional list of specific paper IDs to download. If None, downloads all papers for the date.",
    )


class RankPapersRequest(BaseModel):
    date: str = Field(
        description="Date to rank papers for (format: YYYY-MM-DD).",
        examples=["2024-01-15"],
    )


@agent.tool_plain
@log_tool_usage
async def get_papers() -> str:
    """The user may ask about papers. This feature is not implemented yet."""
    return "We can't handle papers yet - this feature is coming soon."


# @agent.tool_plain
# @log_tool_usage
# async def get_papers(request: GetPapersRequest) -> str:
#     """Check available papers from the past week.

#     Returns a summary of dates and paper counts. Use this to see what papers
#     are already downloaded before asking the user to select dates.
#     """
#     available = await get_available_dates(days_back=request.days_back)
#     return format_available_dates(available)


# @agent.tool_plain
# @log_tool_usage
# async def fetch_papers(request: FetchPapersRequest) -> str:
#     """Fetch paper metadata from HuggingFace for a date (does NOT download).

#     Use this to see what papers are available on HuggingFace for a given date.
#     Returns paper titles and IDs. Does not save anything to disk.
#     """
#     papers = await fetch_papers_for_date(request.date)

#     if not papers:
#         return f"No papers found on HuggingFace for {request.date}"

#     lines = [f"Found {len(papers)} papers for {request.date}:\n"]
#     for p in papers[:20]:  # Limit to first 20 for readability
#         lines.append(f"  - [{p['id']}] {p['title'][:60]}...")

#     if len(papers) > 20:
#         lines.append(f"\n  ... and {len(papers) - 20} more")

#     return "\n".join(lines)


# @agent.tool_plain
# @log_tool_usage
# async def download_papers(request: DownloadPapersRequest) -> str:
#     """Download papers from HuggingFace to local disk.

#     Saves paper metadata to ~/.dxtr/papers/{date}/. Only use this if
#     get_papers shows the papers aren't already downloaded.

#     PREREQUISITE: Call get_papers first to check what's already on disk.
#     """
#     downloaded = await do_download_papers(
#         date=request.date,
#         paper_ids=request.paper_ids,
#         download_pdfs=False,
#     )

#     if not downloaded:
#         return f"No papers downloaded for {request.date}"

#     return f"Downloaded {len(downloaded)} papers for {request.date}"


# @agent.tool_plain
# @log_tool_usage
# async def rank_papers(request: RankPapersRequest) -> str:
#     """Rank papers for a date against the user's synthesized profile.

#     PREREQUISITE: Call get_papers first to verify papers are downloaded.
#     """
#     # Load user profile
#     # Note: This call is known to be missing user_id in the original code.
#     # Leaving it as is but awaiting it to match async definition.
#     # It will likely raise TypeError at runtime if called.
#     profile = await load_profile()
#     if "No synthesized profile found" in profile:
#         return profile

#     # Load papers and convert to dict
#     papers_list = await load_papers_metadata(request.date)
#     if not papers_list:
#         return f"No papers found for {request.date}. Use download_papers first."

#     papers_dict = papers_list_to_dict(papers_list)

#     # Rank papers in parallel
#     results = await papers_ranking.rank_papers_parallel(profile, papers_dict)

#     # Format results
#     rankings_text = format_ranking_results(results)

#     return f"Ranked {len(results)} papers\n\n{rankings_text}"
