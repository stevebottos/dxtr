"""Papers ranking agent - ranks papers by upvotes, profile, or custom request."""

import hashlib
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


def _hash_profile(profile: str) -> str:
    """Deterministic hash of profile text for cache lookup."""
    return hashlib.sha256(profile.strip().encode()).hexdigest()[:16]


def _request_similarity(a: str, b: str) -> float:
    """Simple word overlap similarity between two requests."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)  # Jaccard similarity


def _find_similar_request(
    ctx: RunContext[data_models.PapersRankDeps],
    new_request: str,
    threshold: float = 0.6,
) -> str | None:
    """Find a similar existing request criteria. Returns the criteria if found."""
    db = ctx.deps.db
    table = db.rankings_table

    # Get distinct request criteria for this user/date
    rows = db.query(
        f"""
        SELECT DISTINCT ranking_criteria
        FROM {table}
        WHERE user_id = %s AND paper_date = %s AND ranking_criteria_type = 'request'
        """,
        (ctx.deps.user_id, ctx.deps.date_to_rank),
    )

    for row in rows:
        existing = row["ranking_criteria"]
        if _request_similarity(new_request, existing) >= threshold:
            return existing

    return None


def _store_rankings(
    ctx: RunContext[data_models.PapersRankDeps],
    scored_papers: list[ScoredPaper],
    criteria_type: str,
    criteria: str,
    criteria_hash: str | None,
) -> None:
    """Store scored papers to the rankings table."""
    db = ctx.deps.db
    table = db.rankings_table
    for paper in scored_papers:
        db.execute(
            f"""
            INSERT INTO {table}
            (user_id, paper_id, paper_date, ranking_criteria_type, ranking_criteria, ranking_criteria_hash, ranking, reason)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (
                ctx.deps.user_id,
                paper["id"],
                ctx.deps.date_to_rank,
                criteria_type,
                criteria,
                criteria_hash,
                paper["score"],
                paper["reason"],
            ),
        )


def _get_cached_rankings(
    ctx: RunContext[data_models.PapersRankDeps],
    criteria_type: str,
    criteria_hash: str | None = None,
    criteria_text: str | None = None,
) -> list[dict] | None:
    """Retrieve cached rankings from DB. Returns None if not found."""
    db = ctx.deps.db
    table = db.rankings_table

    if criteria_type == "profile" and criteria_hash:
        rows = db.query(
            f"""
            SELECT r.paper_id, r.ranking, r.reason, p.title, p.summary, p.authors, p.upvotes
            FROM {table} r
            JOIN papers p ON r.paper_id = p.id
            WHERE r.user_id = %s AND r.paper_date = %s
              AND r.ranking_criteria_type = %s AND r.ranking_criteria_hash = %s
            ORDER BY r.ranking DESC
            """,
            (ctx.deps.user_id, ctx.deps.date_to_rank, criteria_type, criteria_hash),
        )
        return rows if rows else None

    if criteria_type == "request" and criteria_text:
        rows = db.query(
            f"""
            SELECT r.paper_id, r.ranking, r.reason, p.title, p.summary, p.authors, p.upvotes
            FROM {table} r
            JOIN papers p ON r.paper_id = p.id
            WHERE r.user_id = %s AND r.paper_date = %s
              AND r.ranking_criteria_type = %s AND r.ranking_criteria = %s
            ORDER BY r.ranking DESC
            """,
            (ctx.deps.user_id, ctx.deps.date_to_rank, criteria_type, criteria_text),
        )
        return rows if rows else None

    return None


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


def _format_summary(papers: list[dict] | list[ScoredPaper], description: str) -> str:
    """Format summary + top 5 abstracts for master agent."""
    total = len(papers)
    top_5 = papers[:5]

    lines = [f"Completed {description}. Ranked {total} papers."]
    lines.append("")
    lines.append("Top 5 papers:")
    for i, p in enumerate(top_5, 1):
        score = p.get("score") or p.get("ranking", "?")
        lines.append(f"{i}. [{score}/5] {p['title']}")
        lines.append(f"   Reason: {p['reason']}")
        summary = p.get("summary", "")
        if summary:
            lines.append(f"   Abstract: {summary}")
        lines.append("")

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
        return "No profile found. Suggest the user either: (1) ask for papers on a specific topic, or (2) chat about their interests first so you can learn about them."

    profile_hash = _hash_profile(ctx.deps.user_profile)

    # Check for cached rankings
    cached = _get_cached_rankings(ctx, "profile", profile_hash)
    if cached:
        print(f"Found cached profile rankings for {ctx.deps.date_to_rank}")
        return _format_summary(cached, f"profile-based rankings for {ctx.deps.date_to_rank}")

    # Compute new rankings
    papers = _get_papers(ctx)
    if not papers:
        return f"No papers found for {ctx.deps.date_to_rank}."

    print(f"Ranking {len(papers)} papers by profile...")
    scored = await _score_papers(papers, f"User Profile:\n{ctx.deps.user_profile}")

    # Store to DB
    _store_rankings(ctx, scored, "profile", ctx.deps.user_profile, profile_hash)

    return _format_summary(scored, f"profile-based rankings for {ctx.deps.date_to_rank}")


@papers_agent.tool
async def rank_by_request(
    ctx: RunContext[data_models.PapersRankDeps],
    request: str = Field(description="What the user is specifically looking for"),
) -> str:
    """Rank papers by relevance to a specific request/topic.

    Use when user asks for papers about a specific topic or question.
    """
    send_internal("tool", f"Ranking papers by request: {request[:50]}...")

    # Check for similar existing request
    similar_criteria = _find_similar_request(ctx, request)
    if similar_criteria:
        cached = _get_cached_rankings(ctx, "request", criteria_text=similar_criteria)
        if cached:
            print(f"Found similar cached request: {similar_criteria[:50]}...")
            return _format_summary(cached, f"request-based rankings for '{request[:30]}...'")

    # Compute new rankings
    papers = _get_papers(ctx)
    if not papers:
        return f"No papers found for {ctx.deps.date_to_rank}."

    print(f"Ranking {len(papers)} papers by request: {request[:50]}...")
    scored = await _score_papers(papers, f"User is looking for:\n{request}")

    # Store to DB (no hash for request-based)
    _store_rankings(ctx, scored, "request", request, None)

    return _format_summary(scored, f"request-based rankings for '{request[:30]}...'")
