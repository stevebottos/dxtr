"""
Papers Ranking Agent

Ranks research papers by relevance to user profile and interests.
Uses a map-reduce pattern:
1. Map: Score each paper individually (parallel)
2. Reduce: Aggregate scores and produce final ranking
"""

import json
from pathlib import Path

import sglang as sgl

from dxtr.agents.base import BaseAgent
from dxtr.config_v2 import config


# Tool definition for main chat agent
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "rank_papers",
        "description": "Rank research papers by relevance to the user's profile and interests.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_query": {
                    "type": "string",
                    "description": "The user's original question/request about papers",
                },
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format (optional, defaults to today)",
                },
            },
            "required": ["user_query"],
        },
    },
}


class Agent(BaseAgent):
    """Agent for ranking research papers using SGLang."""

    def __init__(self):
        super().__init__()
        self.temperature = config.agents.papers_helper_temperature
        self.max_tokens = config.agents.papers_helper_max_tokens
        self.prompts_dir = Path(__file__).parent
        self.score_prompt = self.load_system_prompt(self.prompts_dir / "score.md")
        self.rank_prompt = self.load_system_prompt(self.prompts_dir / "rank.md")

    @staticmethod
    @sgl.function
    def score_paper_func(s, user_message, system_prompt, max_tokens, temp, num_forks=3):
        """Score a single paper with forked reasoning paths for self-consistency."""
        s += sgl.system(system_prompt)
        s += sgl.user(user_message)

        # Fork into multiple reasoning paths from shared prefix
        forks = s.fork(num_forks)
        for i, f in enumerate(forks):
            f += sgl.assistant(
                sgl.gen(f"score_{i}", max_tokens=max_tokens, temperature=temp)
            )

    @staticmethod
    @sgl.function
    def rank_papers_func(s, user_message, system_prompt, max_tokens, temp):
        """Aggregate scores and produce final ranking."""
        s += sgl.system(system_prompt)
        s += sgl.user(user_message)
        s += sgl.assistant(
            sgl.gen("ranking", max_tokens=max_tokens, temperature=temp)
        )

    def _load_papers(self, date: str, papers_dir: Path = None) -> list[dict]:
        """Load paper metadata for a given date."""
        if papers_dir is None:
            papers_dir = config.paths.papers_dir

        date_dir = papers_dir / date
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
                except Exception:
                    continue

        return papers

    def _build_score_message(self, user_context: str, paper: dict) -> str:
        """Build message for scoring a single paper."""
        return f"""# User Profile & Interests

{user_context}

---

# Paper to Evaluate

ID: {paper.get('id')}
Title: {paper.get('title')}
Upvotes: {paper.get('upvotes', 0)}

Abstract:
{paper.get('summary', 'No abstract available.')}"""

    def _build_rank_message(
        self, user_context: str, user_query: str, paper_scores: list[dict]
    ) -> str:
        """Build message for final ranking aggregation."""
        scores_text = "\n\n---\n\n".join([
            f"**Paper ID: {ps['id']}**\n"
            f"Title: {ps['title']}\n\n"
            f"Individual Assessment:\n{ps['score']}"
            for ps in paper_scores
        ])

        return f"""# User's Question

"{user_query}"

---

# User Profile Summary

{user_context[:2000]}  # Truncated for reduce step

---

# Individual Paper Assessments

{scores_text}"""

    def run(
        self,
        date: str,
        user_context: str,
        user_query: str,
        papers_dir: Path = None,
    ) -> dict:
        """
        Rank papers using map-reduce pattern.

        1. Map: Score each paper individually (parallel)
        2. Reduce: Aggregate and produce final ranking
        """
        papers = self._load_papers(date, papers_dir)

        if not papers:
            return {"error": f"No papers found for {date}"}

        # === MAP PHASE: Score each paper in parallel ===
        batch_data = []
        for paper in papers:
            batch_data.append({
                "user_message": self._build_score_message(user_context, paper),
                "system_prompt": self.score_prompt,
                "max_tokens": self.max_tokens,
                "temp": self.temperature,
            })

        states = self.score_paper_func.run_batch(
            batch_data, num_threads=len(papers), progress_bar=True
        )

        # Collect scores
        paper_scores = []
        for i, state in enumerate(states):
            paper_scores.append({
                "id": papers[i].get("id"),
                "title": papers[i].get("title"),
                "score": state["score"],
            })

        # === REDUCE PHASE: Aggregate and rank ===
        rank_message = self._build_rank_message(user_context, user_query, paper_scores)

        rank_state = self.rank_papers_func.run(
            user_message=rank_message,
            system_prompt=self.rank_prompt,
            max_tokens=self.max_tokens,
            temp=self.temperature,
        )

        return {
            "paper_count": len(papers),
            "individual_scores": paper_scores,
            "final_ranking": rank_state["ranking"],
        }
