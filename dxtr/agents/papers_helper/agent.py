"""
Papers Helper Agent

Handles paper-related queries:
1. Ranks papers by relevance to user profile
2. Searches through papers for specific topics
3. Provides summaries and recommendations
"""

import json
from pathlib import Path
from dxtr.base import Agent
from dxtr.config import config


class PapersHelperAgent(Agent):
    """Agent for ranking and analyzing research papers."""

    def __init__(self):
        """Initialize papers helper agent."""
        model_config = config.get_model_config("papers_helper")
        super().__init__(
            name="papers_helper",
            model=model_config.name,
            prompts_dir=Path(__file__).parent / "prompts",
            default_options={
                "temperature": model_config.temperature,
                "num_ctx": model_config.context_window,
            }
        )

    def _load_papers(self, date: str, papers_root: Path = None) -> list[dict]:
        """
        Load paper metadata for a given date.

        Args:
            date: Date string in YYYY-MM-DD format
            papers_root: Root directory where papers are stored

        Returns:
            List of paper metadata dicts
        """
        if papers_root is None:
            papers_root = config.paths.papers_dir

        date_dir = papers_root / date
        if not date_dir.exists():
            return []

        papers = []
        for paper_dir in date_dir.iterdir():
            if not paper_dir.is_dir():
                continue

            metadata_file = paper_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    metadata = json.loads(metadata_file.read_text())
                    papers.append(metadata)
                except Exception as e:
                    print(f"Error loading {metadata_file}: {e}")
                    continue

        return papers

    def rank_papers(self, date: str, user_context: str) -> str:
        """
        Rank papers by relevance to user profile.

        Args:
            date: Date string in YYYY-MM-DD format
            user_context: User profile and GitHub analysis context

        Returns:
            Formatted string with ranked papers and reasoning
        """
        print(f"\n[Papers Helper Agent: Ranking papers for {date}]")

        papers = self._load_papers(date)

        if not papers:
            return f"No papers found for {date}"

        print(f"  [Found {len(papers)} paper(s)]")

        papers_data = []
        for paper in papers:
            papers_data.append({
                "id": paper.get("id"),
                "title": paper.get("title"),
                "upvotes": paper.get("upvotes", 0),
                "summary": paper.get("summary", ""),
            })

        user_prompt = f"""# User Context

{user_context}

---

# Papers to Rank

{json.dumps(papers_data, indent=2)}

Rank these papers from 1-5 (where 5 is most relevant) based on the user's profile and interests. For each paper, provide:
- Relevance score (1-5)
- Brief reasoning (2-3 sentences explaining why it's relevant or not)
- **IMPORTANT**: Include a direct quote from the paper's summary/abstract to support your reasoning

Format your response as a ranked list, starting with the most relevant papers."""

        print("  [Analyzing papers...]")

        response = self.chat(
            messages=[{"role": "user", "content": user_prompt}],
            prompt_name="ranking"
        )

        result = response.message.content
        print(f"  [✓] Papers Helper Agent: Ranking complete")

        return result


# Global instance for backward compatibility
_agent = PapersHelperAgent()


def rank_papers(date: str, user_context: str, papers_root: Path = None) -> str:
    """
    Rank papers by relevance to user profile.

    This is a convenience function that delegates to the agent instance.

    Args:
        date: Date string in YYYY-MM-DD format
        user_context: User profile and GitHub analysis context
        papers_root: Ignored (uses config path)

    Returns:
        Formatted string with ranked papers and reasoning
    """
    return _agent.rank_papers(date, user_context)
