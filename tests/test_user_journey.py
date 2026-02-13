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


HALLUCINATION_SENTINEL = "__HALLUCINATION_CHECK__"


async def _pick_absent_paper_query() -> str:
    """Pick the lowest-scored paper from the rankings — guaranteed outside the top 5 summary."""
    assert MOCK_DB._rankings, "No rankings stored — rank case must run first"

    # Sort by score ascending, pick the lowest
    by_score = sorted(MOCK_DB._rankings, key=lambda r: r["ranking"])
    worst = by_score[0]

    # Find the title from the fixture
    ranked_date = worst["paper_date"]
    date_papers = MOCK_DB._papers.get(ranked_date, [])
    for p in date_papers:
        if p["id"] == worst["paper_id"]:
            return f"Why did '{p['title']}' rank so low?"

    raise RuntimeError(f"Could not find paper {worst['paper_id']} in fixture")


JOURNEY_CASES = [
    Case(
        name="store_profile_facts",
        inputs="""Hi dxtr, here's some info about me: I'm a machine learning engineer with 7 years of experience. Also, right now
        I'm more of an applied AI scientist focused on using foundation models in systems, more engineering-focused than research.""",
        evaluators=[
            ValidateToolBehaviour(tool_fn_name="store_user_fact"),
            ValidateToolBehaviour(
                tool_fn_name="invoke_papers_agent", tool_call_wanted=False
            ),
        ],
    ),
    Case(
        name="store_interest",
        inputs="Oh, also, I'm interested in small language models (SLMs) especially",
        evaluators=[
            ValidateToolBehaviour(tool_fn_name="store_user_fact"),
            ValidateToolBehaviour(
                tool_fn_name="invoke_papers_agent", tool_call_wanted=False
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
        inputs="Check what our most recently saved papers are (we don't save papers on weekends), and rank them for me.",
        evaluators=[
            ValidateToolBehaviour(tool_fn_name="invoke_papers_agent"),
            ValidateToolBehaviour(tool_fn_name="rank"),
            MaxPapersInHistory(
                store=MOCK_STORE,
                session_key=(TEST_USER_ID, TEST_SESSION_ID),
                max_titles=5,
            ),
        ],
    ),
    Case(
        name="hallucination_check",
        inputs=HALLUCINATION_SENTINEL,
        evaluators=[
            ValidateToolBehaviour(tool_fn_name="invoke_papers_agent"),
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
        if inp == HALLUCINATION_SENTINEL:
            inp = await _pick_absent_paper_query()
            print(f"\n=== HALLUCINATION CHECK INPUT: {inp}")
        res = await handle_query(
            MasterRequest(user_id=TEST_USER_ID, session_id=TEST_SESSION_ID, query=inp),
            db=MOCK_DB,
            store=MOCK_STORE,
        )
        if case.name == "hallucination_check":
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
