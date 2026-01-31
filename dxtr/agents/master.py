from pathlib import Path
from tempfile import TemporaryDirectory

import psycopg2
from pydantic import Field
from pydantic_ai import Agent, RunContext

from dxtr import (
    constants,
    data_models,
    get_model_settings,
    get_session_state,
    load_system_prompt,
    log_tool_usage,
    master,
    run_agent,
    update_session_state,
    util,
)
from dxtr.agents.subagents import github_summarizer, papers_ranking, profile_synthesis
from dxtr.agents.util import format_ranking_results
from dxtr.db import PostgresHelper
from dxtr.storage import get_store, get_session_key, create_and_store_artifact

SYSTEM_PROMPT_BASE = load_system_prompt(Path(__file__).parent / "system.md")


# Agent outputs plain text. Artifacts are displayed via display_artifact tool.
agent = Agent(
    master,
    deps_type=data_models.MasterRequest,
    output_type=str,
)


@agent.system_prompt
def build_system_prompt(ctx: RunContext[data_models.MasterRequest]) -> str:
    """Build system prompt with current user state and profile injected."""
    from datetime import datetime

    state = get_session_state()
    today = datetime.now(constants.PST).strftime("%Y-%m-%d")

    # Profile section - full content or "not created"
    if state.profile_content:
        profile_section = f"""
# User Profile
The user's profile is loaded below. Use this to answer questions about them or for personalization.
Do NOT call get_user_profile - the profile is already here.

{state.profile_content}
"""
    else:
        profile_section = """
# User Profile
Not yet created. If the user wants personalized recommendations, help them create a profile first.
"""

    state_section = f"""
# Current State
- Today's date: {today}
- Profile exists: {state.has_synthesized_profile}
- GitHub summary exists: {state.has_github_summary}
"""

    # Artifacts section - shows what's been computed this session
    artifacts_section = state.get_artifact_prompt_section()
    if artifacts_section:
        artifacts_section = "\n" + artifacts_section

    return SYSTEM_PROMPT_BASE + state_section + profile_section + artifacts_section


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

    # Update session state so subsequent tools know github summary exists
    update_session_state(has_github_summary=True)

    return result.output


def extract_chat_only(messages: list) -> str:
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
    """Synthesize an enriched user profile from the conversation context.

    Prerequisites:
    - User has provided background, interests, and goals in the conversation
    - If GitHub links were provided, create_github_summary should be called first

    Call this when you have enough information to create a useful profile.
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

    # Update session state so subsequent tools know profile exists and have content
    update_session_state(has_synthesized_profile=True, profile_content=result.output)

    return result.output


# === Artifact Tools ===


@agent.tool
@log_tool_usage
async def rank_daily_papers(
    ctx: RunContext[data_models.MasterRequest],
    date: str = Field(description="Date to rank papers for (YYYY-MM-DD format)"),
    ranking_type: str = Field(
        default="profile",
        description="Type of ranking: 'profile' (personalized) or 'upvotes' (popularity)",
    ),
) -> str:
    """Compute paper rankings for a date and store as an artifact.

    Creates a cached artifact. After calling this, include the artifact_id in your
    response to display the rankings to the user.

    Only call when:
    - User has a profile (for ranking_type='profile')
    - User wants recommendations for a specific date
    """
    # Validate date format
    data_models.validate_date_format(date)

    session_key = get_session_key(ctx.deps.user_id, ctx.deps.session_id)

    # Get profile from session state (required for profile-based ranking)
    state = get_session_state()
    if ranking_type == "profile" and not state.profile_content:
        return "No profile found. Please create a profile first before ranking papers."
    profile = state.profile_content

    # Load papers from database
    try:
        db = PostgresHelper()
        papers_dict = db.get_papers_for_ranking(date)
    except psycopg2.Error as e:
        return f"Database error loading papers: {e}"

    if not papers_dict:
        return f"No papers found for {date}. Use get_available_papers to check available dates."

    if len(papers_dict) > constants.MAX_PAPERS_TO_RANK:
        return f"Too many papers ({len(papers_dict)}) to rank. Maximum is {constants.MAX_PAPERS_TO_RANK}."

    # Rank papers
    print(f"Ranking {len(papers_dict)} papers for user {ctx.deps.user_id}...")
    results = await papers_ranking.rank_papers_parallel(profile, papers_dict)
    rankings_content = format_ranking_results(results)

    # Create artifact
    summary = f"rankings from {date} based on {ranking_type}"
    artifact_id = await create_and_store_artifact(
        session_key=session_key,
        content=rankings_content,
        summary=summary,
        artifact_type="rankings",
    )

    # Update session state with new artifact registry
    store = get_store()
    updated_state = await store.get_state(session_key)
    update_session_state(artifact_registry=updated_state.artifact_registry,
                         next_artifact_id=updated_state.next_artifact_id)

    # Return summary for LLM context (not full content)
    # Extract top papers for the summary
    top_papers = results[:3] if len(results) >= 3 else results
    top_summary = ", ".join([f"{r['title'][:40]}... ({r['score']}/10)" for r in top_papers])

    return f"Ranked {len(results)} papers (artifact_id={artifact_id}). Top 3: {top_summary}"


@agent.tool
@log_tool_usage
async def read_artifact(
    ctx: RunContext[data_models.MasterRequest],
    artifact_id: int = Field(description="ID of the artifact to read"),
) -> str:
    """Load an artifact's content into context for discussion.

    Use this when you need to answer questions ABOUT the artifact, like:
    - "What makes paper #3 relevant to me?"
    - "Compare these two papers"
    - "Explain the top recommendation"

    This loads the full content into your context so you can discuss it.
    """
    session_key = get_session_key(ctx.deps.user_id, ctx.deps.session_id)
    store = get_store()

    artifact = await store.get_artifact(session_key, artifact_id)
    if not artifact:
        return f"Artifact {artifact_id} not found. Use the artifact IDs from Available Artifacts above."

    return f"Content of artifact {artifact_id} ({artifact.meta.summary}):\n\n{artifact.content}"


@agent.tool
@log_tool_usage
async def display_artifact(
    ctx: RunContext[data_models.MasterRequest],
    choice: int = Field(description="Artifact choice to display. See Available Artifacts in system prompt."),
) -> str:
    """Display an artifact to the user.

    Call this when you want to SHOW an artifact to the user (e.g., rankings).
    The artifact content will be included in the response automatically.

    This does NOT load content into your context. Use read_artifact() if you
    need to discuss or analyze the content.
    """
    session_key = get_session_key(ctx.deps.user_id, ctx.deps.session_id)
    store = get_store()

    artifact = await store.get_artifact(session_key, choice)
    if not artifact:
        return f"Artifact {choice} not found. Check Available Artifacts above for valid choices."

    # Queue for display - server will include content in response
    state = get_session_state()
    state.queue_for_display(choice)
    update_session_state(pending_display_artifacts=state.pending_display_artifacts)

    return f"Artifact {choice} ({artifact.meta.summary}) queued for display."


# === Paper Tools ===


@agent.tool_plain
@log_tool_usage
async def get_available_papers(request: data_models.GetPapersRequest) -> str:
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
    except psycopg2.Error as e:
        return f"Database error: {e}"


@agent.tool_plain
@log_tool_usage
async def get_paper_stats(request: data_models.GetPaperStatsRequest) -> str:
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
    except psycopg2.Error as e:
        return f"Database error: {e}"


@agent.tool_plain
@log_tool_usage
async def get_top_papers(request: data_models.GetTopPapersRequest) -> str:
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
    except psycopg2.Error as e:
        return f"Database error: {e}"


