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
    """
    send_internal("tool", "Storing user fact...")
    db = ctx.deps.db
    fact_id = db.execute_returning(
        f"INSERT INTO {db.facts_table} (user_id, fact) VALUES (%s, %s) RETURNING id",
        (ctx.deps.request.user_id, fact),
    )
    return f"Stored fact (id={fact_id})"


@agent.tool
async def invoke_papers_rank_agent(
    ctx: RunContext[data_models.AgentDeps],
    date_to_rank: str = Field(description="Date of papers to rank (YYYY-MM-DD format)"),
    query: str = Field(description="The user's original request about papers"),
) -> str:
    """Delegate paper ranking to the papers agent.

    Use this when the user asks for paper recommendations or rankings.
    The papers agent will decide the best ranking method.
    """
    send_internal("tool", f"Ranking papers for {date_to_rank}...")
    deps = data_models.PapersRankDeps(
        user_id=ctx.deps.request.user_id,
        date_to_rank=date_to_rank,
        user_profile=ctx.deps.context.user_profile_facts,
        papers_by_date=ctx.deps.context.papers_by_date,
        db=ctx.deps.db,
    )
    result = await papers_agent.run(query, deps=deps)
    return result.output
