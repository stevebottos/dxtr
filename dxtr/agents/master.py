from pathlib import Path

from pydantic import Field
from pydantic_ai import Agent, RunContext
from pydantic_ai_litellm import LiteLLMModel


from dxtr import constants, data_models, load_system_prompt
from dxtr.agents.subagents.papers_ranking.agent import papers_agent
from dxtr.bus import send_internal

SYSTEM_PROMPT_BASE = load_system_prompt(Path(__file__).parent / "system.md")

agent = Agent(
    LiteLLMModel(
        model_name="openai/master",
        api_key=constants.API_KEY,
        api_base=constants.BASE_URL,
    ),
    deps_type=data_models.AgentDeps,
    output_type=str,
    system_prompt=SYSTEM_PROMPT_BASE,
)


@agent.system_prompt
async def add_user_context(ctx: RunContext[data_models.AgentDeps]):
    """Inject user profile facts and today's date from pre-fetched context."""
    context = ctx.deps.context
    return f"Today's date: {context.today_date}\n\n{context.user_profile_facts}"


@agent.tool
async def store_user_fact(
    ctx: RunContext[data_models.AgentDeps],
    fact: str = Field(
        description="A meaningful fact about the user (background, interests, goals, expertise, preferences)"
    ),
) -> str:
    """Store a fact learned about the user during conversation.

    Use this when you learn something meaningful about the user that would be
    useful for personalization.

    Do NOT store transient conversation details or trivial observations.
    Do NOT store redundant facts.
    Only store facts when the user explicitly shares something about themselves.
    Do NOT store facts derived from their ranking requests.
    """
    send_internal("tool", "Storing user fact...")
    db = ctx.deps.db
    fact_id = db.execute_returning(
        f"INSERT INTO {db.facts_table} (user_id, fact) VALUES (%s, %s) RETURNING id",
        (ctx.deps.request.user_id, fact),
    )
    return f"Stored fact (id={fact_id})"


@agent.tool
async def invoke_papers_agent(
    ctx: RunContext[data_models.AgentDeps],
    date: str = Field(
        description="Date of papers (YYYY-MM-DD format). Use the date reference table in context to resolve relative dates like 'today', 'yesterday', etc."
    ),
    query: str = Field(
        description="What to do — e.g. 'Rank papers' or 'Compare paper X and paper Y'"
    ),
) -> str:
    """Rank or compare papers for a given date. This is a "by request" tool.

    Call this when the user asks to rank, recommend, or compare papers they haven't seen yet.
    Also call this when the user asks about a specific paper that is NOT in your conversation context — use it to retrieve details rather than guessing.
    Do NOT call this for follow-up questions about papers already visible in context (e.g. "tell me more about the top one", "which was highest?").
    Do NOT call this just because the user mentions a topic — store that as a fact instead.
    """
    send_internal("tool", f"Papers agent: {query}")
    deps = data_models.PapersRankDeps(
        user_id=ctx.deps.request.user_id,
        date_to_rank=date,
        user_profile=ctx.deps.context.user_profile_facts,
        papers_by_date=ctx.deps.context.papers_by_date,
        db=ctx.deps.db,
    )
    result = await papers_agent.run(query, deps=deps)
    return result.output
