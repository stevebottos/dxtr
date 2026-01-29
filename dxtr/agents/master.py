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
    load_user_profile,
)
from dxtr.db import PostgresHelper

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
async def get_available_papers(request: GetPapersRequest) -> str:
    """Check what dates have papers available in our database.

    Returns a summary of dates and paper counts from the past N days.
    Use this to see what papers are available before fetching or ranking.

    NOTE: For aggregate queries like "how many papers total" or "what's the busiest day",
    use get_paper_stats instead - it computes totals directly.
    """
    try:
        db = PostgresHelper()
        available = db.get_available_dates(days_back=request.days_back)

        if not available:
            return "No papers found in database. Papers may need to be fetched."

        lines = ["Available papers in database:"]
        for item in available:
            lines.append(f"  {item['date']}: {item['count']} papers")

        return "\n".join(lines)
    except Exception as e:
        return f"Error querying database: {e}"


class GetPaperStatsRequest(BaseModel):
    days_back: int = Field(
        default=7,
        description="Number of days to look back for statistics. Use a larger value (e.g., 30, 365) for broader queries.",
    )


@agent.tool_plain
@log_tool_usage
async def get_paper_stats(request: GetPaperStatsRequest) -> str:
    """Get aggregate statistics about papers in the database.

    Returns pre-computed totals and averages - use this for questions like:
    - "How many papers total do we have?"
    - "How many papers from last week?"
    - "What's the average upvotes?"
    - "Which day had the most papers?"

    This is more efficient than get_available_papers for aggregate queries
    because it computes totals directly rather than listing each day.
    """
    try:
        db = PostgresHelper()
        stats = db.get_paper_stats(days_back=request.days_back)

        if not stats or stats.get("total_papers", 0) == 0:
            return f"No papers found in the last {request.days_back} days."

        # Also get the busiest day
        busiest = db.get_date_with_most_papers(days_back=request.days_back)
        busiest_line = ""
        if busiest:
            busiest_line = f"\n  Busiest day: {busiest['date']} ({busiest['count']} papers)"

        return f"""Paper statistics (last {request.days_back} days):
  Total papers: {stats['total_papers']}
  Days with papers: {stats['days_with_papers']}
  Date range: {stats['earliest_date']} to {stats['latest_date']}
  Average upvotes: {stats['avg_upvotes']}{busiest_line}"""
    except Exception as e:
        return f"Error querying database: {e}"


class GetTopPapersRequest(BaseModel):
    date: str = Field(
        description="Date to get papers for (format: YYYY-MM-DD).",
        examples=["2024-01-15"],
    )
    limit: int = Field(
        default=10,
        description="Maximum number of papers to return.",
    )


@agent.tool_plain
@log_tool_usage
async def get_top_papers(request: GetTopPapersRequest) -> str:
    """Get the highest upvoted papers for a specific date.

    Returns papers sorted by upvotes (most popular first).
    Use get_available_papers first to see what dates have papers.
    """
    try:
        db = PostgresHelper()
        papers = db.get_top_papers(request.date, request.limit)

        if not papers:
            return f"No papers found for {request.date}. Use get_available_papers to see available dates."

        lines = [f"Top {len(papers)} papers for {request.date} (by upvotes):\n"]
        for p in papers:
            title = p["title"][:60] + "..." if len(p["title"]) > 60 else p["title"]
            lines.append(f"  [{p['upvotes']} upvotes] {title}")
            lines.append(f"    ID: {p['id']}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error querying database: {e}"


@agent.tool
@log_tool_usage
async def rank_papers_for_user(
    ctx: RunContext[data_models.MasterRequest], request: RankPapersRequest
) -> str:
    """Rank papers for a SINGLE SPECIFIC DATE against the user's profile.

    This triggers the ranking agent to score each paper against the user's
    synthesized profile. Papers are scored 1-10 based on relevance.

    IMPORTANT: Only call this when the user EXPLICITLY requests paper ranking.
    Do NOT call proactively after profile creation or other operations.

    CONSTRAINT: Only ranks papers for ONE date at a time (max ~60 papers).
    For queries like "best papers ever" or "all time favorites", explain that
    ranking must be done one day at a time and ask which date to rank.

    PREREQUISITE: User must have a synthesized profile.
    """
    MAX_PAPERS_TO_RANK = 60

    # Load user profile
    profile = await load_user_profile(ctx.deps.user_id)
    if "No synthesized profile found" in profile:
        return profile

    # Load papers from database
    try:
        db = PostgresHelper()
        papers_dict = db.get_papers_for_ranking(request.date)
    except Exception as e:
        return f"Error loading papers from database: {e}"

    if not papers_dict:
        return f"No papers found for {request.date}. Use get_available_papers to see available dates."

    if len(papers_dict) > MAX_PAPERS_TO_RANK:
        return f"Too many papers ({len(papers_dict)}) to rank at once. Maximum is {MAX_PAPERS_TO_RANK}. Please narrow down to a specific date."

    # Rank papers in parallel using the ranking subagent
    print(f"Ranking {len(papers_dict)} papers for user {ctx.deps.user_id}...")
    results = await papers_ranking.rank_papers_parallel(profile, papers_dict)

    # Format results
    rankings_text = format_ranking_results(results)

    return f"Ranked {len(results)} papers for {request.date}\n\n{rankings_text}"
