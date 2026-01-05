"""
Papers Ranking Agent

Ranks research papers by relevance to user profile and interests.
Uses run_batch to parallelize across papers, with forked reasoning per paper.
"""

import json
from pathlib import Path

import sglang as sgl

from dxtr.agents.base import BaseAgent
from dxtr.config_v2 import config


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

# Schema for scoring a single paper
SINGLE_PAPER_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
}

THREAD_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["answer", "reason"],
}


class Agent(BaseAgent):
    """Paper ranking with run_batch parallelization and forked reasoning per paper."""

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

    @staticmethod
    @sgl.function
    def _rank_papers_pre(s, user_context, paper, max_tokens, temp):
        """SGLang function for the final profile enrichment step."""
        s += sgl.system(
            f"The following is a comprehensive profile of the current user. You are a research agent, and must decide whether a research paper is relevant, given this profile: {user_context}."
        )
        s += sgl.user(f"The paper abstract: {paper['summary']}\n")

        forks = s.fork(3)
        # Define different 'personas' or 'strategies' for each fork
        forks[0] += sgl.user(
            "Think step-by-step: what about the user's profile makes this paper relevant? Be specific, include reasoning."
        )
        forks[1] += sgl.user(
            "Is this paper domain specific? Is there any reference to this domain in the user's profile? Include reasoning."
        )
        forks[2] += sgl.user(
            "Consider the current demands in the industry. The user's profile might not necessarily align with the industry's state. How does this paper rank in terms of relevance to the industry as a whole? Include reasoning."
        )

        for i, f in enumerate(forks):
            f += sgl.gen(
                f"reasoning_{i}",
                max_tokens=max_tokens,
                temperature=temp,
                frequency_penalty=1.1,
                presence_penalty=0.1,
                json_schema=json.dumps(THREAD_SCHEMA),
            )

        reasoning_results = []
        for i, f in enumerate(forks):
            # We must await/ensure the generation is captured
            reasoning_results.append(f"Analysis {i + 1}: " + f[f"reasoning_{i}"])

        combined_reasoning = "\n\n".join(reasoning_results)

        # Optional: Join them back to make a final decision
        # We pick the first fork's state to continue, but we can access all
        upvotes = paper["upvotes"]
        s += sgl.user(
            f"""Based on the following reflections: {combined_reasoning}, give a final 1-5 score, where 5 is most relevant, and provide your reasoning. The rubric is as follows:
            Note that a score of 5 is very prestigious. Sometimes, a 5 is a 4.
            As a final check, consider the number of upvotes. This paper has {upvotes} upvotes. If it has more than 100, this is an automatic 5."""
        )
        s += sgl.gen(
            "soft_score",
            max_tokens=max_tokens,
            temperature=0.0,
            json_schema=json.dumps(SINGLE_PAPER_SCORE_SCHEMA),
            frequency_penalty=1.1,
            presence_penalty=0.1,
        )

    def run(
        self,
        date: str,
        user_context: str,
        user_query: str,
        papers_dir: Path = None,
        verbose: bool = True,
    ) -> dict:
        """
        Rank papers using run_batch parallelization with forked reasoning per paper.
        """
        papers = self._load_papers(date, papers_dir)

        if not papers:
            return {"error": f"No papers found for {date}"}

        batch_data = [
            {
                "user_context": user_context,
                "paper": p,
                "max_tokens": 500,
                "temp": 0.0,
            }
            for p in papers
        ]
        states = self._rank_papers_pre.run_batch(
            batch_data, num_threads=128, progress_bar=True
        )

        # Extract results from batch
        results = {}
        for i, state in enumerate(states):
            paper = papers[i]
            paper_id = paper.get("id")

            try:
                output = json.loads(state["soft_score"])
                score = int(output["score"])
                reason = output.get("reason", "")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                score = None
                reason = ""

            results[paper_id] = {
                "title": paper.get("title"),
                "score": score,
                "reason": reason,
            }
            print(reason)

        # Sort by score descending
        sorted_results = dict(
            sorted(results.items(), key=lambda x: x[1]["score"] or 0, reverse=True)
        )

        if verbose:
            print(f"\n{'=' * 60}")
            print("FINAL RANKING")
            print(f"{'=' * 60}")
            for pid, data in sorted_results.items():
                boost_tag = " [BOOST]" if data.get("boosted") else ""
                print(f"{data['score']}/5 - {pid}: {data['title'][:50]}{boost_tag}")

        # Build final ranking string
        ranking_lines = []
        for rank, (pid, data) in enumerate(sorted_results.items(), 1):
            ranking_lines.append(f"{rank}. [{data['score']}/5] {data['title']}")
        final_ranking = "\n".join(ranking_lines)

        # Return in format compatible with eval
        individual_scores = [
            {
                "id": pid,
                "title": data["title"],
                "final_score": data["score"],
                "reason": data.get("reason", ""),
                "boosted": data.get("boosted", False),
            }
            for pid, data in sorted_results.items()
        ]

        return {
            "paper_count": len(papers),
            "individual_scores": individual_scores,
            "scores": {pid: data["score"] for pid, data in sorted_results.items()},
            "final_ranking": final_ranking,
        }
