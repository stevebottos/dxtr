"""
Papers Ranking Agent

Ranks research papers by relevance to user profile and interests.
Uses run_batch to parallelize across papers, with forked reasoning per paper.
"""

import re
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
        "relevance": {"type": "string"},
    },
    "required": ["score", "relevance"],
}

REFLECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "reflection": {"type": "string"},
        "relevance": {"type": "string"},
    },
    "required": ["reflection", "relevance"],
}

SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "value": {"type": "integer"},
    },
    "required": ["value"],
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

        s["paper_id"] = paper["id"]
        s["paper_title"] = paper["title"]
        s["upvotes"] = paper["upvotes"]

        s += sgl.system(
            """
            You are an expert research assistant. Your goal is to rank research papers based on their value to a specific user.

            You will be given:
            1. A User Profile: Contains background, interests, constraints (hardware/compute), and goals.
            2. A Paper Abstract: The summary of a research paper.

            **Scoring Rubric (1-5):**

            **5 (Critical Read):**
            - Perfect alignment with "Current Goals" AND "Emerging/Learning" areas.
            - Directly addresses specific constraints mentioned (e.g., low compute/VRAM, open source).
            - Synergizes multiple interests (e.g., Intersection of CV + LLMs for a multimodal engineer).

            **4 (High Priority):**
            - Strong alignment with core interests or goals.
            - Technically relevant but might miss a specific constraint (e.g., requires high compute but methodology is vital).
            - Excellent resource for "Knowledge Gaps" identified in the profile.

            **3 (Relevant / Good to Know):**
            - Good match for "Strong Areas" (maintenance of expertise) but lacks novelty/connection to new goals.
            - Relevant domain but slightly tangential focus (e.g., hardware-specific when user is software-focused).
            - General architectural improvements without specific application to user's niche.

            **2 (Low Priority / Tangential):**
            - Valid domain but "old news" for the user (e.g., pure CV task they already mastered, with no LLM component).
            - Very niche application unrelated to user's goals (e.g., medical audio, specific robotics hardware).

            **1 (Irrelevant):**
            - Completely different domain (e.g., biology, pure systems/logs) with no clear transferability.

            **Decision Process:**
            1. **Check Constraints:** Does the user have hardware/compute limits? Does this paper help or hurt that?
            2. **Check Goals:** Does this help them move from where they are (Background) to where they want to be (Goals)?
            3. **Check Intersection:** Does it bridge their skills? (e.g., Multimodal > Unimodal).

            Be strict. A "5" is reserved for papers that make the user say "I need to read this now."

            When considering your scoring, you should be terse and direct. Follow this logic stream: Ask, is this paper a 1? Why/Why not?.. A 2? Why/Why not...
            and so on.
            """
        )

        num_forks = 3
        forks = s.fork(num_forks)
        for i in range(num_forks):
            # We use 'forks[i]' to point to a specific parallel branch
            forks[i] += sgl.user(f"The user profile: {user_context}")
            forks[i] += sgl.user(f"The paper abstract: {paper['summary']}")
            forks[i] += sgl.user(
                "Now consider possible scores for this paper, with reasons. Then, settle on a final score."
            )
            forks[i] += sgl.gen(
                "relevance",
                max_tokens=600,
                temperature=0.2,
                frequency_penalty=1.1,
                presence_penalty=0.1,
            )
            forks[i] += (
                "\n---\nFinal Score (1-5), integer only. You must answer in valid json."
            )
            forks[i] += sgl.gen(
                "score",
                max_tokens=20,
                temperature=0,
                json_schema=json.dumps(SCORE_SCHEMA),
            )

        # 3. Retrieve the results
        # Accessing the fork index automatically waits for completion
        reasoning = [f["relevance"] for f in forks]
        scores = [int(json.loads(f["score"])["value"]) for f in forks]

        # Optional: store them back in the main state if needed
        s["relevance"] = reasoning
        s["score"] = scores
        #     "score",
        #     max_tokens=10,
        #     temperature=0.0,
        # )
        # forks = s.fork(3)
        # forks[0] += sgl.user(
        #     "Think step-by-step: what about the user's profile makes this paper relevant? Be specific, include reasoning."
        # )
        # forks[1] += sgl.user(
        #     "Is this paper domain specific? Is there any reference to this domain in the user's profile? Include reasoning."
        # )
        # forks[2] += sgl.user(
        #     "Consider the current demands in the industry. The user's profile might not necessarily align with the industry's state. How does this paper rank in terms of relevance to the industry as a whole? Include reasoning."
        # )
        #
        # for i, f in enumerate(forks):
        #     f += sgl.gen(
        #         f"reasoning_{i}",
        #         max_tokens=max_tokens,
        #         temperature=temp,
        #         frequency_penalty=1.1,
        #         presence_penalty=0.1,
        #         json_schema=json.dumps(THREAD_SCHEMA),
        #     )
        #
        # reasoning_results = []
        # for i, f in enumerate(forks):
        #     # We must await/ensure the generation is captured
        #     reasoning_results.append(f"Analysis {i + 1}: " + f[f"reasoning_{i}"])
        #
        # combined_reasoning = "\n\n".join(reasoning_results)
        #
        # # Optional: Join them back to make a final decision
        # # We pick the first fork's state to continue, but we can access all
        # upvotes = paper["upvotes"]
        # s += sgl.user(
        #     f"""Reflect on your thoughts, are they accurate with regards to the profile's skills and interests: {combined_reasoning}.
        #     Denote relevance as high, medium, low. Note that all papers are relevant, and that low does not imply irrelevant."""
        # )
        # s += sgl.gen(
        #     "reflection",
        #     max_tokens=max_tokens,
        #     temperature=0.0,
        #     json_schema=json.dumps(REFLECTION_SCHEMA),
        #     frequency_penalty=1.1,
        #     presence_penalty=0.1,
        # )
        # s += sgl.user(
        #     f"""Now, finally, assign a 1-5 score to this paper, according to the following rubric:
        #     5. Very relevant, has significant overlap with the profile and the user's career goals, and significant utility
        #     4. Relevant, but not quite a 5. Something must definitely be a 5, if there is any doubt, it's a 4.
        #     3. Some doubt with regard to relevants, it is subjective or situationally relevant
        #     2. Mildly relevant, not high priority
        #     1. Slightly relevant, but if there was a stack of 100 papers to read this wouldn't be in the top 10.
        #
        #     Note on upvotes: If the paper's upvotes ({upvotes})>100, then automatically give 5
        #     """
        # )
        # s += sgl.user("Your score and reasoning?")
        #
        # s += sgl.gen(
        #     "soft_score",
        #     max_tokens=max_tokens,
        #     temperature=0.0,
        #     json_schema=json.dumps(SINGLE_PAPER_SCORE_SCHEMA),
        #     frequency_penalty=1.1,
        #     presence_penalty=0.1,
        # )

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
            print(state["relevance"])
            print(state["score"])
            score = round(sum(state["score"]) / len(state["score"]), 1)
            print(score)

            print("------")
            # match = re.search(r"<score>\s*([1-5])\s*</score>", state["score"])
            # if match:
            #     score = int(match.group(1))
            # else:
            #     score = -1

            # if state["upvotes"] > 100:
            #     score = 5  # Auto score 5 if the paper is popular

            reason = state["relevance"]

            results[paper_id] = {
                "title": state["paper_title"],
                "score": score,
                "reason": reason,
            }
        # Sort by score descending
        sorted_results = dict(
            sorted(results.items(), key=lambda x: x[1]["score"] or 0, reverse=True)
        )

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
