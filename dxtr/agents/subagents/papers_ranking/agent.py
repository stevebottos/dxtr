"""Papers ranking agent - ranks papers by upvotes, profile, or custom request."""

from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai_litellm import LiteLLMModel

from dxtr import constants, data_models, load_system_prompt
from dxtr.agents.subagents.util import parallel_map
from dxtr.bus import send_internal
from dxtr.db import PostgresHelper


# === Data Types ===


class PaperMetadata(TypedDict):
    id: str
    title: str
    summary: str
    authors: list[str]
    upvotes: int


class ScoredPaper(TypedDict):
    id: str
    title: str
    summary: str
    authors: list[str]
    upvotes: int
    score: int  # 1-5
    reason: str


class PaperScore(BaseModel):
    """LLM output for scoring a paper."""

    score: int = Field(ge=1, le=5)
    reason: str = Field(max_length=100)


# === Scoring Agent (scores one paper at a time) ===

scoring_agent = Agent(
    LiteLLMModel(
        model_name="openai/papers_ranker",
        api_key=constants.API_KEY,
        api_base=constants.BASE_URL,
    ),
    system_prompt=load_system_prompt(Path(__file__).parent / "system.md"),
    output_type=PaperScore,
)


# === Papers Ranking Agent ===

papers_agent = Agent(
    LiteLLMModel(
        model_name="openai/papers_ranker",
        api_key=constants.API_KEY,
        api_base=constants.BASE_URL,
    ),
    deps_type=data_models.PapersRankDeps,
    output_type=str,
    system_prompt=load_system_prompt(Path(__file__).parent / "papers_agent_system.md"),
)


# === Helpers ===


def _get_papers(ctx: RunContext[data_models.PapersRankDeps]) -> list[dict]:
    """Get all papers for a date from fixtures or DB."""
    if ctx.deps.papers_by_date is not None:
        return ctx.deps.papers_by_date.get(ctx.deps.date_to_rank, [])

    return ctx.deps.db.query(
        "SELECT id, title, summary, authors, published_at, upvotes, date FROM papers WHERE date = %s ORDER BY upvotes DESC",
        (ctx.deps.date_to_rank,),
    )


def _to_metadata(paper: dict) -> PaperMetadata:
    return PaperMetadata(
        id=paper["id"],
        title=paper["title"],
        summary=paper["summary"],
        authors=paper.get("authors", []),
        upvotes=paper.get("upvotes", 0),
    )


def _format_metadata_list(papers: list[PaperMetadata]) -> str:
    lines = []
    for i, p in enumerate(papers, 1):
        lines.append(f"{i}. [{p['upvotes']} upvotes] {p['title']}")
    return "\n".join(lines)


def _format_scored_list(papers: list[ScoredPaper]) -> str:
    lines = []
    for i, p in enumerate(papers, 1):
        lines.append(f"{i}. [{p['score']}/5] {p['title']}")
        lines.append(f"   {p['reason']}")
    return "\n".join(lines)


async def _score_papers(papers: list[dict], scoring_context: str) -> list[ScoredPaper]:
    """Score papers in parallel using the scoring agent."""

    async def score_one(paper: dict, idx: int, total: int) -> ScoredPaper:
        title = paper["title"]
        short_title = title[:40] + "..." if len(title) > 40 else title
        print(f"  [{idx}/{total}] Scoring: {short_title}", flush=True)

        prompt = f"""## Scoring Context
{scoring_context}

## Paper to Score
**{title}**

{paper.get("summary", "")}
"""
        try:
            result = await scoring_agent.run(prompt)
            print(f"  [{idx}/{total}] Done: {result.output.score}/5", flush=True)
            return ScoredPaper(
                id=paper["id"],
                title=title,
                summary=paper.get("summary", ""),
                authors=paper.get("authors", []),
                upvotes=paper.get("upvotes", 0),
                score=result.output.score,
                reason=result.output.reason,
            )
        except Exception as e:
            print(f"  [{idx}/{total}] Error: {e}", flush=True)
            return ScoredPaper(
                id=paper["id"],
                title=title,
                summary=paper.get("summary", ""),
                authors=paper.get("authors", []),
                upvotes=paper.get("upvotes", 0),
                score=0,
                reason=f"Error: {e}",
            )

    results = await parallel_map(papers, score_one, desc="Scoring papers")
    return sorted(results, key=lambda x: x["score"], reverse=True)


# === Tools ===


@papers_agent.tool
async def rank_by_upvotes(ctx: RunContext[data_models.PapersRankDeps]) -> str:
    """Rank papers by community upvotes (popularity).

    Use when user wants popular papers or doesn't specify personalization.
    """
    send_internal("tool", "Ranking papers by upvotes...")
    papers = _get_papers(ctx)
    if not papers:
        return f"No papers found for {ctx.deps.date_to_rank}."

    metadata = [_to_metadata(p) for p in papers]
    return _format_metadata_list(metadata)


@papers_agent.tool
async def rank_by_profile(ctx: RunContext[data_models.PapersRankDeps]) -> str:
    """Rank papers by relevance to user's profile/interests.

    Use when user wants personalized recommendations based on their background.
    """
    send_internal("tool", "Ranking papers by user profile...")
    if not ctx.deps.user_profile or ctx.deps.user_profile.strip() == "":
        return "No user profile available. Cannot rank by profile."

    papers = _get_papers(ctx)
    if not papers:
        return f"No papers found for {ctx.deps.date_to_rank}."

    print(f"Ranking {len(papers)} papers by profile...")
    scored = await _score_papers(papers, f"User Profile:\n{ctx.deps.user_profile}")
    return _format_scored_list(scored)


@papers_agent.tool
async def rank_by_request(
    ctx: RunContext[data_models.PapersRankDeps],
    request: str = Field(description="What the user is specifically looking for"),
) -> str:
    """Rank papers by relevance to a specific request/topic.

    Use when user asks for papers about a specific topic or question.
    """
    send_internal("tool", f"Ranking papers by request: {request[:50]}...")
    papers = _get_papers(ctx)
    if not papers:
        return f"No papers found for {ctx.deps.date_to_rank}."

    print(f"Ranking {len(papers)} papers by request: {request[:50]}...")
    scored = await _score_papers(papers, f"User is looking for:\n{request}")
    return _format_scored_list(scored)
