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


async def _build_followup_query(query_type: str) -> str:
    """Build a follow-up question from stored rankings data."""
    assert MOCK_DB._rankings, "No rankings stored — rank case must run first"
    by_score = sorted(MOCK_DB._rankings, key=lambda r: r["ranking"])
    papers_idx = MOCK_DB._build_papers_index()

    if query_type == "low_rank":
        worst = by_score[0]
        title = papers_idx[worst["paper_id"]]["title"]
        return f"Why did '{title}' rank so low?"

    if query_type == "compare":
        worst = by_score[0]
        best = by_score[-1]
        t1 = papers_idx[worst["paper_id"]]["title"]
        t2 = papers_idx[best["paper_id"]]["title"]
        return f"Compare '{t1}' and '{t2}'"

    if query_type == "details":
        mid = by_score[len(by_score) // 2]
        title = papers_idx[mid["paper_id"]]["title"]
        return f"Tell me more about '{title}'"

    raise ValueError(f"Unknown query type: {query_type}")


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
            ValidateOutputTool(tool_fn_name="set_rankings"),
            MaxPapersInHistory(
                store=MOCK_STORE,
                session_key=(TEST_USER_ID, TEST_SESSION_ID),
                max_titles=5,
            ),
        ],
    ),
    Case(
        name="hallucination_low_rank",
        inputs=f"{HALLUCINATION_SENTINEL}:low_rank",
        evaluators=[
            ValidateToolBehaviour(tool_fn_name="discuss_papers"),
            ValidateToolBehaviour(tool_fn_name="get_paper_index"),
            ValidateOutputTool(tool_fn_name="set_rankings", tool_call_wanted=False),
            JudgeOutput(
                judge=JUDGE,
                criteria="""The user asked a follow-up question about a specific paper's ranking.
                The agent should answer based on actual ranking data — not make up scores, reasons, or details.
                If the agent says it doesn't know or needs to check, that's acceptable.
                If the agent provides specific scores or reasons, they should sound grounded (not vague or generic).
                Return True if the response seems grounded, False if it appears to hallucinate details.

                Agent response: {agent_output}""",
            ),
        ],
    ),
    Case(
        name="hallucination_compare",
        inputs=f"{HALLUCINATION_SENTINEL}:compare",
        evaluators=[
            ValidateToolBehaviour(tool_fn_name="discuss_papers"),
            ValidateToolBehaviour(tool_fn_name="get_paper_index"),
            ValidateOutputTool(tool_fn_name="set_rankings", tool_call_wanted=False),
            JudgeOutput(
                judge=JUDGE,
                criteria="""The user asked to compare two papers.
                The agent should reference actual differences between the papers — not make up details.
                If the agent provides scores, topics, or reasons, they should sound grounded and specific.
                Return True if the response seems grounded, False if it appears to hallucinate details.

                Agent response: {agent_output}""",
            ),
        ],
    ),
    Case(
        name="hallucination_details",
        inputs=f"{HALLUCINATION_SENTINEL}:details",
        evaluators=[
            ValidateToolBehaviour(tool_fn_name="discuss_papers"),
            ValidateToolBehaviour(tool_fn_name="get_paper_index"),
            ValidateOutputTool(tool_fn_name="set_rankings", tool_call_wanted=False),
            JudgeOutput(
                judge=JUDGE,
                criteria="""The user asked for more details about a specific paper.
                The agent should provide information grounded in the paper's actual abstract and ranking data.
                If the agent provides details about the paper's content, they should match what a real abstract would contain.
                Return True if the response seems grounded, False if it appears to hallucinate details.

                Agent response: {agent_output}""",
            ),
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
