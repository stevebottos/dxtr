"""Papers ranking agent - ranks papers by user profile."""

from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ToolOutput
from pydantic_ai_litellm import LiteLLMModel

from dxtr import constants, data_models, load_system_prompt
from dxtr.agents.subagents.util import parallel_map
from dxtr.bus import send_internal


# === Data Types ===


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


# === Helpers ===


def _store_rankings(
    ctx: RunContext[data_models.PapersRankDeps],
    scored_papers: list[ScoredPaper],
    criteria: str,
    date: str,
) -> None:
    """Store scored papers to the rankings table."""
    db = ctx.deps.db
    table = db.rankings_table
    for paper in scored_papers:
        db.execute(
            f"""
            INSERT INTO {table}
            (user_id, paper_id, paper_date, ranking_criteria_type, ranking_criteria, ranking, reason)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (
                ctx.deps.user_id,
                paper["id"],
                date,
                "profile",
                criteria,
                paper["score"],
                paper["reason"],
            ),
        )


def _get_papers(ctx: RunContext[data_models.PapersRankDeps], date: str) -> list[dict]:
    """Get all papers for a date from fixtures or DB."""
    if ctx.deps.papers_by_date is not None:
        return ctx.deps.papers_by_date.get(date, [])

    return ctx.deps.db.query(
        "SELECT id, title, summary, authors, published_at, upvotes, date FROM papers WHERE date = %s ORDER BY upvotes DESC",
        (date,),
    )


def _format_summary(papers: list[ScoredPaper], description: str) -> str:
    """Format a lean summary of top 5 papers for master agent.

    No abstracts — the master only needs titles/scores/reasons to present results.
    Full details are fetched on demand via get_paper_index + get_paper_details.
    """
    total = len(papers)
    top_5 = papers[:5]

    lines = [f"Completed {description}. Ranked {total} papers."]
    lines.append("")
    lines.append("Top 5 papers:")
    for i, p in enumerate(top_5, 1):
        lines.append(f"{i}. [{p['score']}/5] {p['title']}")
        lines.append(f"   Reason: {p['reason']}")
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
    failed = [r for r in results if r["score"] == 0]
    scored = [r for r in results if r["score"] > 0]

    if failed:
        titles = [f["title"][:50] for f in failed]
        send_internal(
            "warning", f"Failed to score {len(failed)}/{len(results)} papers: {titles}"
        )

    return sorted(scored, key=lambda x: x["score"], reverse=True)


# === Output Tools ===


async def set_rankings(
    ctx: RunContext[data_models.PapersRankDeps],
    date: str = Field(description="Date of papers to rank (YYYY-MM-DD format)"),
) -> str:
    """Score and rank all papers for a date by relevance to the user's profile."""
    profile = ctx.deps.user_profile
    if not profile or not profile.strip():
        return "No user profile available. Suggest the user chat about their interests first so you can build a profile."

    send_internal("tool", "Ranking papers by profile...")

    papers = _get_papers(ctx, date)
    if not papers:
        return f"No papers found for {date}."

    print(f"Ranking {len(papers)} papers by user profile...")
    scored = await _score_papers(papers, f"User Profile:\n{profile}")

    _store_rankings(ctx, scored, profile, date)

    return _format_summary(scored, "profile-based ranking")


# === Papers Ranking Agent ===

papers_agent = Agent(
    LiteLLMModel(
        model_name="openai/papers_ranker",
        api_key=constants.API_KEY,
        api_base=constants.BASE_URL,
    ),
    deps_type=data_models.PapersRankDeps,
    output_type=[ToolOutput(set_rankings, name='set_rankings'), str],
    system_prompt=load_system_prompt(Path(__file__).parent / "papers_agent_system.md"),
)


@papers_agent.system_prompt
async def add_context(ctx: RunContext[data_models.PapersRankDeps]) -> str:
    """Inject date reference table and ranked dates into the papers agent context."""
    lines = [ctx.deps.today_date]

    if ctx.deps.ranked_dates:
        ranked = ", ".join(ctx.deps.ranked_dates)
        lines.append(f"\nDates with existing rankings: {ranked}")
        lines.append("For these dates, use `get_paper_index` + `get_paper_details` to answer questions — do NOT call `set_rankings`.")
    else:
        lines.append("\nNo dates have been ranked yet.")

    return "\n".join(lines)


@papers_agent.tool
async def get_paper_index(
    ctx: RunContext[data_models.PapersRankDeps],
    date: str = Field(description="Date of rankings to look up (YYYY-MM-DD format)"),
) -> str:
    """Get a lightweight index of all ranked papers for a date.

    Returns paper ID, title, score, and reason — no abstracts or authors.
    Use this first to identify which papers the user is asking about,
    then call get_paper_details with the specific paper IDs you need.
    """
    send_internal("tool", "Retrieving paper index...")
    db = ctx.deps.db
    rankings = db.query(
        f"""SELECT r.paper_id, r.ranking, r.reason, p.title
            FROM {db.rankings_table} r
            JOIN papers p ON r.paper_id = p.id
            WHERE r.user_id = %s AND r.paper_date = %s
              AND r.ranking_criteria_type = 'profile'
            ORDER BY r.ranking DESC""",
        (ctx.deps.user_id, date),
    )
    if not rankings:
        return "No rankings found for this date."

    lines = [f"{len(rankings)} ranked papers:"]
    for r in rankings:
        lines.append(f"- [{r['ranking']}/5] {r['title']} (ID: {r['paper_id']}) — {r['reason']}")
    return "\n".join(lines)


@papers_agent.tool
async def get_paper_details(
    ctx: RunContext[data_models.PapersRankDeps],
    paper_ids: list[str],
    date: str = Field(description="Date of rankings to look up (YYYY-MM-DD format)"),
) -> str:
    """Get full details for specific papers by their IDs.

    Returns abstract, authors, upvotes, score, and reason for each paper.
    Only call this after using get_paper_index to identify the relevant paper IDs.
    """
    send_internal("tool", "Retrieving paper details...")
    db = ctx.deps.db
    results = []
    for paper_id in paper_ids:
        paper_rows = db.query(
            "SELECT id, title, summary, authors, upvotes FROM papers WHERE id = %s",
            (paper_id,),
        )
        ranking_rows = db.query(
            f"""SELECT r.ranking, r.reason
                FROM {db.rankings_table} r
                WHERE r.user_id = %s AND r.paper_id = %s AND r.paper_date = %s
                  AND r.ranking_criteria_type = 'profile'""",
            (ctx.deps.user_id, paper_id, date),
        )
        if not paper_rows:
            results.append(f"Paper {paper_id}: not found")
            continue
        p = paper_rows[0]
        r = ranking_rows[0] if ranking_rows else {}
        authors = p.get("authors") or []
        author_names = [a["name"] if isinstance(a, dict) else a for a in authors]
        lines = [
            f"**{p['title']}** (ID: {p['id']})",
            f"  Score: {r.get('ranking', 'N/A')}/5 | Reason: {r.get('reason', 'N/A')}",
            f"  Authors: {', '.join(author_names) if author_names else 'N/A'}",
            f"  Upvotes: {p.get('upvotes', 0)}",
            f"  Abstract: {p.get('summary', 'N/A')}",
        ]
        results.append("\n".join(lines))
    return "\n\n".join(results)
