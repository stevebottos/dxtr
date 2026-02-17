from dataclasses import dataclass

import logfire
import pytest
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai_litellm import LiteLLMModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from dxtr import constants
from dxtr.data_models import MasterRequest
from dxtr.server import handle_query
from tests.mocks import InMemoryDB, InMemoryConversationStore

logfire.configure(send_to_logfire=True)
logfire.instrument_pydantic_ai()

# === Test Configuration ===

TEST_USER_ID = "dev_user_steve"
TEST_SESSION_ID = "papers_test_session"
MOCK_DB = InMemoryDB()
MOCK_STORE = InMemoryConversationStore()
LATEST_PAPER_DATE = max(MOCK_DB._papers.keys())


class JudgeResult(BaseModel):
    passed: bool = Field(
        description="True if the agent's response meets the criteria, False otherwise"
    )
    reasoning: str = Field(
        description="Brief explanation of why the response passed or failed"
    )


JUDGE = Agent(
    LiteLLMModel(
        model_name="openai/judge",
        api_key=constants.API_KEY,
        api_base=constants.BASE_URL,
    ),
    output_type=JudgeResult,
    system_prompt="You are a judge evaluating agent output quality. The pass/fail criteria will be provided to you. Return your boolean verdict and a brief reasoning.",
    retries=5,
)


@dataclass
class ValidateToolBehaviour(Evaluator):
    tool_fn_name: str
    tool_call_wanted: bool = True
    tool_call_key: str = "gen_ai.tool.name"

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        tool_calls = ctx.span_tree.find(lambda s: s.name == "running tool")

        for tool_call in tool_calls:
            if tool_call.attributes[self.tool_call_key] == self.tool_fn_name:
                return self.tool_call_wanted
        return not self.tool_call_wanted


@dataclass
class ValidateMultiPaperRetrieval(Evaluator):
    """Check that get_paper_details was called with at least min_papers paper IDs."""

    min_papers: int = 2
    tool_name_key: str = "gen_ai.tool.name"
    tool_args_key: str = "tool_arguments"

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        tool_calls = ctx.span_tree.find(lambda s: s.name == "running tool")

        for tool_call in tool_calls:
            if tool_call.attributes.get(self.tool_name_key) == "get_paper_details":
                raw_args = tool_call.attributes.get(self.tool_args_key, "")
                import json
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except (json.JSONDecodeError, TypeError):
                    continue
                paper_ids = args.get("paper_ids", [])
                if len(paper_ids) >= self.min_papers:
                    return True
        return False


@dataclass
class ValidateOutputTool(Evaluator):
    """Check that a tool was executed as an output tool (exits the agent run directly)."""

    tool_fn_name: str
    tool_call_wanted: bool = True
    tool_call_key: str = "gen_ai.tool.name"

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        output_tools = ctx.span_tree.find(
            lambda s: s.name == "running output function"
        )
        for span in output_tools:
            if span.attributes.get(self.tool_call_key) == self.tool_fn_name:
                return self.tool_call_wanted
        return not self.tool_call_wanted


@dataclass
class JudgeOutput(Evaluator):
    criteria: str
    judge: Agent

    async def evaluate(self, ctx: EvaluatorContext) -> bool:
        agent_output = ctx.output.output
        query = self.criteria.format(agent_output=agent_output)

        res = await self.judge.run(query)
        print(res.output.reasoning)
        return res.output.passed


@dataclass
class GroundedResponseEvaluator(Evaluator):
    """Judge the agent's response against ground-truth paper data from the mock DB."""

    query_type: str
    db: InMemoryDB
    judge: Agent

    async def evaluate(self, ctx: EvaluatorContext) -> bool:
        papers = _resolve_papers_for_query(self.query_type, self.db)

        # Build ground truth reference block
        reference_lines = []
        for p in papers:
            raw_authors = p.get("authors", [])
            if raw_authors and isinstance(raw_authors[0], dict):
                authors = ", ".join(a.get("name", str(a)) for a in raw_authors)
            else:
                authors = ", ".join(str(a) for a in raw_authors)
            reference_lines.append(
                f"- Title: {p['title']}\n"
                f"  Abstract: {p['summary']}\n"
                f"  Authors: {authors}\n"
                f"  Score: {p['ranking']}/5\n"
                f"  Reason: {p['reason']}"
            )
        reference = "\n".join(reference_lines)

        prompt = (
            f"## Reference Data (ground truth)\n{reference}\n\n"
            f"## Agent Response\n{ctx.output.output}\n\n"
            f"## Task\nDoes the agent's response accurately reflect the reference data above? "
            f"Flag any fabricated details — invented authors, wrong scores, made-up abstract content, "
            f"or claims not supported by the reference. Minor rephrasing is fine. "
            f"Return True if grounded, False if it contains hallucinated details."
        )

        res = await self.judge.run(prompt)
        print(f"  [GroundedResponseEvaluator:{self.query_type}] {res.output.reasoning}")
        return res.output.passed


@dataclass
class RankingQualityEvaluator(Evaluator):
    """Judge whether rankings are well-differentiated and aligned with the user profile."""

    db: InMemoryDB
    judge: Agent

    async def evaluate(self, ctx: EvaluatorContext) -> bool:
        if not self.db._rankings:
            print("  [RankingQualityEvaluator] No rankings stored — skipping")
            return False

        # Gather user profile
        profile_facts = [f["fact"] for f in self.db._facts]
        profile_str = "\n".join(f"- {fact}" for fact in profile_facts) if profile_facts else "No profile stored."

        # Gather rankings with paper info
        papers_idx = self.db._build_papers_index()
        ranking_lines = []
        for r in sorted(self.db._rankings, key=lambda x: x["ranking"], reverse=True):
            p = papers_idx.get(r["paper_id"], {})
            ranking_lines.append(
                f"- [{r['ranking']}/5] {p.get('title', r['paper_id'])}\n"
                f"  Abstract: {p.get('summary', 'N/A')}\n"
                f"  Reason: {r['reason']}"
            )
        rankings_str = "\n".join(ranking_lines)

        # Score distribution summary
        scores = [r["ranking"] for r in self.db._rankings]
        distribution = {s: scores.count(s) for s in sorted(set(scores))}
        dist_str = ", ".join(f"{s}/5: {c} papers" for s, c in distribution.items())

        prompt = (
            f"## User Profile\n{profile_str}\n\n"
            f"## Score Distribution\n{dist_str}\n\n"
            f"## Rankings ({len(self.db._rankings)} papers)\n{rankings_str}\n\n"
            f"## Task\nEvaluate whether these rankings are reasonable:\n"
            f"1. **Differentiation**: Are the scores well-spread across the 1-5 range? "
            f"If most papers got the same score (e.g. all 5/5 or all 3/5), that's a failure — "
            f"papers on different topics should receive different relevance scores.\n"
            f"2. **Profile alignment**: Do higher-scored papers genuinely match the user's stated interests better than lower-scored ones?\n"
            f"3. **Reason quality**: Are the reasons specific to each paper, or are they generic/copy-pasted?\n\n"
            f"Return True only if the rankings are meaningfully differentiated AND aligned with the profile. "
            f"Return False if the scores are clustered (poor differentiation) or misaligned with the profile."
        )

        res = await self.judge.run(prompt)
        print(f"  [RankingQualityEvaluator] {res.output.reasoning}")
        return res.output.passed


@dataclass
class MaxPapersInHistory(Evaluator):
    """Check that the conversation history doesn't contain more than `max_titles` paper titles."""

    store: InMemoryConversationStore
    session_key: tuple[str, str]
    max_titles: int = 5

    async def evaluate(self, ctx: EvaluatorContext) -> bool:
        history = await self.store.get_history(self.session_key)

        # Collect all text content from the conversation history
        full_text = ""
        for msg in history:
            for part in msg.parts:
                if hasattr(part, "content") and isinstance(part.content, str):
                    full_text += part.content + "\n"

        # Get all paper titles from the fixture for the date the agent chose
        all_titles = [p["title"] for papers in MOCK_DB._papers.values() for p in papers]
        found = [t for t in all_titles if t in full_text]

        print(
            f"Found {len(found)}/{len(all_titles)} paper titles in history (max allowed: {self.max_titles})"
        )
        return len(found) <= self.max_titles


HALLUCINATION_SENTINEL = "__HALLUCINATION__"


def _resolve_papers_for_query(query_type: str, db: InMemoryDB) -> list[dict]:
    """Resolve which paper(s) a query type targets, returning full paper data with ranking info."""
    assert db._rankings, "No rankings stored — rank case must run first"
    by_score = sorted(db._rankings, key=lambda r: r["ranking"])
    papers_idx = db._build_papers_index()

    def _enrich(ranking_row: dict) -> dict:
        p = papers_idx[ranking_row["paper_id"]]
        return {
            "paper_id": ranking_row["paper_id"],
            "title": p["title"],
            "summary": p.get("summary", ""),
            "authors": p.get("authors", []),
            "ranking": ranking_row["ranking"],
            "reason": ranking_row["reason"],
        }

    if query_type == "low_rank":
        return [_enrich(by_score[0])]

    if query_type == "compare":
        return [_enrich(by_score[0]), _enrich(by_score[-1])]

    if query_type == "details":
        mid = by_score[len(by_score) // 2]
        return [_enrich(mid)]

    if query_type == "outside_top5":
        by_score_desc = list(reversed(by_score))
        assert len(by_score_desc) > 5, "Need >5 ranked papers for outside_top5 test"
        return [_enrich(by_score_desc[5])]

    raise ValueError(f"Unknown query type: {query_type}")


async def _build_followup_query(query_type: str) -> str:
    """Build a follow-up question from stored rankings data."""
    papers = _resolve_papers_for_query(query_type, MOCK_DB)

    if query_type == "low_rank":
        return f"Why did '{papers[0]['title']}' rank so low?"

    if query_type == "compare":
        return f"Compare '{papers[0]['title']}' and '{papers[1]['title']}'"

    if query_type == "details":
        return f"Tell me more about '{papers[0]['title']}'"

    if query_type == "outside_top5":
        return f"Tell me about '{papers[0]['title']}'"

    raise ValueError(f"Unknown query type: {query_type}")


JOURNEY_CASES = [
    Case(
        name="store_profile_facts",
        inputs="""Hi dxtr, here's some info about me: I'm a machine learning engineer with 7 years of experience. Also, right now
        I'm more of an applied AI scientist focused on using foundation models in systems, more engineering-focused than research.""",
        evaluators=[
            ValidateToolBehaviour(tool_fn_name="store_user_fact"),
            ValidateToolBehaviour(
                tool_fn_name="ask_papers_agent", tool_call_wanted=False
            ),
        ],
    ),
    Case(
        name="store_interest",
        inputs="Oh, also, I'm interested in small language models (SLMs) especially",
        evaluators=[
            ValidateToolBehaviour(tool_fn_name="store_user_fact"),
            ValidateToolBehaviour(
                tool_fn_name="ask_papers_agent", tool_call_wanted=False
            ),
        ],
    ),
    Case(
        name="recall_facts",
        inputs="Tell me what you know about me so far.",
        evaluators=[
            JudgeOutput(
                judge=JUDGE,
                criteria="""We have previously mentioned:
                - I'm a machine learning engineer with 7 years of experience.
                - I'm more of an applied AI scientist focused on using foundation models in systems, more engineering-focused than research.
                - I'm interested in small language models (SLMs) especially

                We are testing the agent's ability to retain this information. Its response should contain this information. You should return True if it does.

                This is the agent's response that we're testing: {agent_output}""",
            )
        ],
    ),
    Case(
        name="rank_papers",
        inputs=f"Rank papers for {LATEST_PAPER_DATE}",
        evaluators=[
            ValidateToolBehaviour(tool_fn_name="ask_papers_agent"),
            ValidateOutputTool(tool_fn_name="set_rankings"),
            MaxPapersInHistory(
                store=MOCK_STORE,
                session_key=(TEST_USER_ID, TEST_SESSION_ID),
                max_titles=5,
            ),
            RankingQualityEvaluator(db=MOCK_DB, judge=JUDGE),
        ],
    ),
    Case(
        name="hallucination_low_rank",
        inputs=f"{HALLUCINATION_SENTINEL}:low_rank",
        evaluators=[
            ValidateToolBehaviour(tool_fn_name="ask_papers_agent"),
            ValidateToolBehaviour(tool_fn_name="get_paper_index"),
            ValidateOutputTool(tool_fn_name="set_rankings", tool_call_wanted=False),
            GroundedResponseEvaluator(query_type="low_rank", db=MOCK_DB, judge=JUDGE),
        ],
    ),
    Case(
        name="hallucination_compare",
        inputs=f"{HALLUCINATION_SENTINEL}:compare",
        evaluators=[
            ValidateToolBehaviour(tool_fn_name="ask_papers_agent"),
            ValidateToolBehaviour(tool_fn_name="get_paper_index"),
            ValidateMultiPaperRetrieval(min_papers=2),
            ValidateOutputTool(tool_fn_name="set_rankings", tool_call_wanted=False),
            GroundedResponseEvaluator(query_type="compare", db=MOCK_DB, judge=JUDGE),
        ],
    ),
    Case(
        name="hallucination_details",
        inputs=f"{HALLUCINATION_SENTINEL}:details",
        evaluators=[
            ValidateToolBehaviour(tool_fn_name="ask_papers_agent"),
            ValidateToolBehaviour(tool_fn_name="get_paper_index"),
            ValidateOutputTool(tool_fn_name="set_rankings", tool_call_wanted=False),
            GroundedResponseEvaluator(query_type="details", db=MOCK_DB, judge=JUDGE),
        ],
    ),
    Case(
        name="hallucination_outside_top5",
        inputs=f"{HALLUCINATION_SENTINEL}:outside_top5",
        evaluators=[
            ValidateToolBehaviour(tool_fn_name="ask_papers_agent"),
            ValidateToolBehaviour(tool_fn_name="get_paper_index"),
            ValidateOutputTool(tool_fn_name="set_rankings", tool_call_wanted=False),
            GroundedResponseEvaluator(query_type="outside_top5", db=MOCK_DB, judge=JUDGE),
        ],
    ),
]


@pytest.fixture(scope="module", autouse=True)
async def cleanup_database():
    """Reset in-memory stores before tests in this module."""
    MOCK_DB.reset()
    MOCK_STORE.reset()
    yield


@pytest.mark.asyncio
@pytest.mark.parametrize("case", JOURNEY_CASES, ids=[c.name for c in JOURNEY_CASES])
async def test_user_journey(case: Case):
    async def run_snapshot(inp):
        if inp.startswith(HALLUCINATION_SENTINEL):
            query_type = inp.split(":", 1)[1]
            inp = await _build_followup_query(query_type)
            print(f"\n=== HALLUCINATION CHECK ({query_type}): {inp}")
        res = await handle_query(
            MasterRequest(user_id=TEST_USER_ID, session_id=TEST_SESSION_ID, query=inp),
            db=MOCK_DB,
            store=MOCK_STORE,
        )
        if case.name.startswith("hallucination_"):
            print(f"=== AGENT OUTPUT: {res.output}\n")
        return res

    dataset = Dataset(cases=[case])
    report = await dataset.evaluate(run_snapshot)
    print(report)

    assert not report.failures, f"Task execution failed: {report.failures}"
    for rc in report.cases:
        failed = [name for name, result in rc.assertions.items() if not result.value]
        assert not failed, f"Case '{rc.name}' failed evaluators: {failed}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
