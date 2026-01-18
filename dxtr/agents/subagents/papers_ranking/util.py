"""Utilities for the papers ranking agent."""

from dxtr import DXTR_DIR


def load_profile() -> str:
    """Load the user's synthesized profile."""
    profile_path = DXTR_DIR / "synthesized_profile.md"
    if not profile_path.exists():
        return "No synthesized profile found. Create one first."
    return profile_path.read_text()


def papers_list_to_dict(papers: list[dict]) -> dict[str, dict]:
    """Convert list of papers to dict keyed by ID."""
    return {
        p["id"]: {"title": p.get("title", ""), "summary": p.get("summary", "")}
        for p in papers
    }


def format_ranking_results(results: list[dict]) -> str:
    """Format ranking results for display.

    Args:
        results: List of {id, title, score, reason} dicts, sorted by score

    Returns:
        Formatted markdown string
    """
    if not results:
        return "No papers ranked."

    lines = ["# Paper Rankings", ""]

    current_tier = None
    for r in results:
        score = r["score"]

        # Determine tier
        if score >= 9:
            tier = "Must Read (9-10)"
        elif score >= 7:
            tier = "Highly Relevant (7-8)"
        elif score >= 5:
            tier = "Moderately Relevant (5-6)"
        elif score >= 3:
            tier = "Low Relevance (3-4)"
        else:
            tier = "Not Relevant (1-2)"

        if tier != current_tier:
            current_tier = tier
            lines.append(f"## {tier}")
            lines.append("")

        lines.append(f"**[{score}/10]** {r['title']}")
        lines.append(f"  - {r['reason']}")
        lines.append(f"  - `{r['id']}`")
        lines.append("")

    return "\n".join(lines)
