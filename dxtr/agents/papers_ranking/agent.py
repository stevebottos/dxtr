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

# Schema for critic adjustments
CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "adjustments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "original_score": {"type": "integer"},
                    "adjusted_score": {"type": "integer"},
                    "issue": {"type": "string"},
                },
                "required": ["id", "original_score", "adjusted_score", "issue"],
            },
        },
    },
    "required": ["adjustments"],
}

# Evaluation criteria for Tree of Thought (3 branches that merge)
TOT_CRITERIA = [
    ("relevance", "How relevant is this paper to the user's stated interests?"),
    ("practical", "What practical value does this paper have for the user's work?"),
    (
        "alignment",
        "Does this paper align with topics the user has expressed interest in?",
    ),
]


class Agent(BaseAgent):
    """Paper ranking with run_batch parallelization and forked reasoning per paper."""

    def __init__(self):
        super().__init__()
        # self.score_prompt = self.load_system_prompt(self.prompts_dir / "score.md")
        # self.rank_prompt = self.load_system_prompt(self.prompts_dir / "rank.md")
        # self.critic_prompt = self.load_system_prompt(self.prompts_dir / "critic.md")
        # self.system_prompt = self.load_system_prompt(
        #     Path(__file__).parent / "system.md"
        # )
        self.system_prompt = (
            None  # Being explicit here. System prompts are created on the fly.
        )

    @staticmethod
    @sgl.function
    def score_all_papers_tot(s, user_context, papers, system_prompt, criteria):
        """Score all papers using Tree of Thought.

        Architecture:
        - Base context: system prompt + user profile (shared, ~400 tokens)
        - Fork per paper: each adds just abstract (~300 tokens)
        - Fork per criterion: 3 criteria evaluated in parallel
        - MERGE: Fork outputs merged back, final score sees all reasoning

        This is true ToT - the final decision sees the parallel explorations.
        """
        # Shared base context - cached via radix
        s += sgl.system(system_prompt)
        s += sgl.user(f"# User Profile\n\n{user_context}\n\n---\n\n")

        # Fork once per paper
        paper_forks = s.fork(len(papers))

        for pi, pf in enumerate(paper_forks):
            paper = papers[pi]
            # Each paper fork adds just the abstract
            pf += sgl.user(
                f"# Paper to Evaluate\n\n"
                f"**{paper.get('id')}**: {paper.get('title')}\n"
                f"Upvotes: {paper.get('upvotes', 0)}\n\n"
                f"{paper.get('summary', 'No abstract.')}"
            )

            # Fork into criteria branches (Tree of Thought)
            forks = pf.fork(len(criteria))
            for f, (name, question) in zip(forks, criteria):
                f += sgl.assistant(question + " ")
                f += sgl.gen(name, max_tokens=150, temperature=0.0, stop="\n\n")

            # MERGE: Fork outputs back into main context
            pf += sgl.assistant("Based on my analysis:\n")
            for f, (name, _) in zip(forks, criteria):
                pf += f"- {name.title()}: " + f[name] + "\n"

            # Final score generation sees all merged reasoning
            pf += "\nFinal assessment: "
            pf += sgl.gen(
                "final_score",
                max_tokens=200,
                temperature=0.0,
                json_schema=json.dumps(SINGLE_PAPER_SCORE_SCHEMA),
            )

        # Collect final scores from paper forks
        for pi, pf in enumerate(paper_forks):
            s[f"paper_{pi}_score"] = pf["final_score"]

        s["num_papers"] = len(papers)

    @staticmethod
    @sgl.function
    def rank_papers_func(s, user_message, system_prompt, max_tokens, temp):
        """Produce final ranking from aggregated scores."""
        s += sgl.system(system_prompt)
        s += sgl.user(user_message)
        s += sgl.assistant(sgl.gen("ranking", max_tokens=max_tokens, temperature=temp))

    @staticmethod
    @sgl.function
    def critic_func(s, user_message, system_prompt):
        """Review scores and reasoning, suggest adjustments for factual errors."""
        s += sgl.system(system_prompt)
        s += sgl.user(user_message)
        s += sgl.assistant(
            sgl.gen(
                "critique",
                max_tokens=2000,
                temperature=0.0,
                json_schema=json.dumps(CRITIC_SCHEMA),
            )
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

    def _build_single_paper_context(self, user_context: str, paper: dict) -> str:
        """Build context for a single paper evaluation."""
        return f"""# User Profile

{user_context}

---

# Paper to Evaluate

**{paper.get("id")}**: {paper.get("title")}
Upvotes: {paper.get("upvotes", 0)}

{paper.get("summary", "No abstract.")}"""

    def _build_critic_context(self, user_context: str, results: dict) -> str:
        """Build context for the critic to review all scores and reasoning."""
        lines = [
            "# User Profile",
            "",
            user_context,
            "",
            "---",
            "",
            "# Scored Papers",
            "",
        ]

        for paper_id, data in results.items():
            lines.append(f"## {paper_id}: {data['title']}")
            lines.append(f"Score: {data['score']}/5")
            reason = data.get("reason", "")
            if reason:
                lines.append(f"Reasoning: {reason}")
            lines.append("")

        return "\n".join(lines)

    def _apply_critic_adjustments(
        self, results: dict, adjustments: list, verbose: bool
    ) -> dict:
        """Apply critic adjustments to scores."""
        for adj in adjustments:
            paper_id = adj.get("id")
            if paper_id in results:
                old_score = results[paper_id]["score"]
                new_score = adj.get("adjusted_score")
                issue = adj.get("issue", "")

                if verbose:
                    print(f"  CRITIC: {paper_id} {old_score} -> {new_score}: {issue}")

                results[paper_id]["score"] = new_score
                results[paper_id]["critic_adjusted"] = True
                results[paper_id]["critic_issue"] = issue

        return results

    def _aggregate_fork_scores(self, state, num_forks: int, upvotes: int = 0) -> dict:
        """Aggregate scores from forks for a single paper, with upvote boost."""
        scores = []
        reasons = []

        for i in range(num_forks):
            try:
                output = json.loads(state[f"fork_{i}"])
                scores.append(int(output["score"]))
                reasons.append(output.get("reason", ""))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

        if not scores:
            return {"score": None, "scores": [], "reasons": [], "boosted": False}

        avg_score = int(round(sum(scores) / len(scores)))

        # Apply upvote boost: 100+ upvotes = 5, 50+ upvotes = at least 4
        boosted = False
        if upvotes >= 100:
            if avg_score < 5:
                avg_score = 5
                boosted = True
        elif upvotes >= 50:
            if avg_score < 4:
                avg_score = 4
                boosted = True

        return {
            "score": avg_score,
            "scores": scores,
            "reasons": reasons,
            "boosted": boosted,
        }

    @staticmethod
    @sgl.function
    def _rank_papers_pre(s, user_context, paper, max_tokens, temp):
        """SGLang function for the final profile enrichment step."""
        s += sgl.system(
            f"The following is a comprehensive profile of the current user. You are a research agent, and must decide whether a research paper is relevant, given this profile: {user_context}."
        )
        s += sgl.user(f"The paper info: {paper}")

        forks = s.fork(3)
        upvotes = paper["upvotes"]
        # Define different 'personas' or 'strategies' for each fork
        forks[0] += sgl.user(
            "Think step-by-step: what about the user's profile makes this paper relevant? Be specific."
        )
        forks[1] += sgl.user(
            "Is this paper domain specific? Is there any reference to this domain in the user's profile?"
        )
        forks[2] += sgl.user(
            "Not all papers can be read, there is limited time in the day. If I had 10 papers to read this week, would I fast track this one?"
        )

        # Generate reasoning for all forks in parallel
        for i, f in enumerate(forks):
            f += sgl.gen(f"reasoning_{i}", max_tokens=max_tokens, temperature=temp)

        # Optional: Join them back to make a final decision
        # We pick the first fork's state to continue, but we can access all
        s += sgl.user(
            f"""Based on the reasoning above, give a final 1-5 score, where 5 is most relevant, and provide your reasoning. The rubric is as follows:
            5 - Very relevant, prioritize this paper, read it today
            4 - Relevant, read this paper this week
            3 - Read this paper if there are no 4/5 scoring ones in the reading list
            2 - This paper is most likely irrelevant, read it if you are curious 
            1 - This paper is definitely irrelevant, save your time
            
            Note that a 5 score is prestigious. There needs to be a very valid reason to assign a 5. Assume that this is one paper out of 20 that you need to rank, 
            and not all should get 5's. Some 5's are really 4's.
            As a final check, consider the number of upvotes. This paper has {upvotes} upvotes. If it has more than 100, this is an automatic 5."""
        )
        s += sgl.gen(
            "soft_score",
            max_tokens=max_tokens,
            temperature=temp,
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

        mt = 250
        temp = 0.1
        batch_data = [
            {
                "user_context": user_context,
                "paper": p,
                "max_tokens": mt,
                "temp": temp,
            }
            for p in papers
        ]
        states = self._rank_papers_pre.run_batch(
            batch_data, num_threads=128, progress_bar=True
        )

        results = []
        for i, state in enumerate(states):
            score = state["soft_score"]
            print(score)

        exit()
        state = self._rank_papers_pre.run(
            user_context=user_context,
            papers=papers,
            system_prompt=self.score_prompt,
            criteria=TOT_CRITERIA,
        )

        # Extract results from ToT structure
        results = {}

        for pi, paper in enumerate(papers):
            paper_id = paper.get("id")
            upvotes = paper.get("upvotes", 0)

            try:
                output = json.loads(state[f"paper_{pi}_score"])
                score = int(output["score"])
                reason = output.get("reason", "")
            except (json.JSONDecodeError, KeyError, TypeError):
                score = None
                reason = ""

            # Apply upvote boost
            boosted = False
            if score is not None:
                if upvotes >= 100 and score < 5:
                    score = 5
                    boosted = True
                elif upvotes >= 50 and score < 4:
                    score = 4
                    boosted = True

            results[paper_id] = {
                "title": paper.get("title"),
                "score": score,
                "reason": reason,
                "boosted": boosted,
            }

            if verbose:
                boost_tag = " [BOOSTED]" if boosted else ""
                print(f"\n{paper_id}: {score}/5{boost_tag}")
                print(f"  Title: {paper.get('title', '')[:60]}")
                print(f"  Reason: {reason[:100]}{'...' if len(reason) > 100 else ''}")

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
        }
