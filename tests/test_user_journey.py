from dataclasses import dataclass
from datetime import date

import logfire
import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai_litellm import LiteLLMModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from dxtr import constants
from dxtr.data_models import MasterRequest
from dxtr.db import PostgresHelper
from dxtr.server import dev_nuke_redis, handle_query

logfire.configure(send_to_logfire=True)
logfire.instrument_pydantic_ai()

# === Test Configuration ===

TEST_USER_ID = "dev_user_steve"
TEST_SESSION_ID = "papers_test_session"
TODAY = date.today().isoformat()
DEV_DB = PostgresHelper(is_dev=True)
TODAYS_PAPERS = [
    {
        "id": "2601.20614",
        "title": "Harder Is Better: Boosting Mathematical Reasoning via Difficulty-Aware GRPO",
        "summary": "Reinforcement Learning with Verifiable Rewards (RLVR) offers a robust mechanism for enhancing mathematical reasoning in large models.",
        "authors": ["Yanqi Dai", "Yuxiang Ji"],
        "upvotes": 93,
    },
    {
        "id": "2601.20540",
        "title": "Advancing Open-source World Models",
        "summary": "We present LingBot-World, an open-sourced world simulator stemming from video generation.",
        "authors": ["Robbyant Team", "Zelin Gao"],
        "upvotes": 65,
    },
    {
        "id": "2601.19325",
        "title": "Innovator-VL: A Multimodal Large Language Model for Scientific Discovery",
        "summary": "We present Innovator-VL, a scientific multimodal large language model designed to advance understanding and reasoning across diverse scientific domains.",
        "authors": ["Zichen Wen", "Boxue Yang"],
        "upvotes": 53,
    },
    {
        "id": "2601.20552",
        "title": "DeepSeek-OCR 2: Visual Causal Flow",
        "summary": "We present DeepSeek-OCR 2 to investigate the feasibility of a novel encoder capable of dynamically reordering visual tokens.",
        "authors": ["Haoran Wei", "Yaofeng Sun"],
        "upvotes": 25,
    },
    {
        "id": "2601.20209",
        "title": "Spark: Strategic Policy-Aware Exploration for Long-Horizon Agentic Learning",
        "summary": "Reinforcement learning has empowered large language models to act as intelligent agents for long-horizon tasks.",
        "authors": ["Jinyang Wu", "Shuo Yang"],
        "upvotes": 12,
    },
    {
        "id": "2601.20834",
        "title": "Linear representations in language models can change dramatically over a conversation",
        "summary": "Language model representations often contain linear directions that correspond to high-level concepts.",
        "authors": ["Andrew Kyle Lampinen", "Yuxuan Li"],
        "upvotes": 8,
    },
    {
        "id": "2601.20802",
        "title": "Reinforcement Learning via Self-Distillation",
        "summary": "Large language models are increasingly post-trained with reinforcement learning in verifiable domains.",
        "authors": ["Jonas Hübotter", "Frederike Lübeck"],
        "upvotes": 5,
    },
    {
        "id": "2601.20055",
        "title": "VERGE: Formal Refinement and Guidance Engine for Verifiable LLM Reasoning",
        "summary": "We present a neurosymbolic framework that combines LLMs with SMT solvers for verification-guided answers.",
        "authors": ["Vikash Singh", "Darion Cassel"],
        "upvotes": 5,
    },
]


class JudgeResult(BaseModel):
    passed: bool
    reasoning: str


JUDGE = Agent(
    LiteLLMModel(
        model_name="openai/judge",
        api_key=constants.API_KEY,
        api_base=constants.BASE_URL,
    ),
    output_type=JudgeResult,
    system_prompt="You are acting as a judge, marking evaluation quality. The pass/fail criteria will be provided to you. You are to simply output a boolean, True if passed.",
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


JOURNEY_CASES = [
    Case(
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
        inputs="Oh, also, I'm interested in small language models (SLMs) especially",
        evaluators=[
            ValidateToolBehaviour(tool_fn_name="store_user_fact"),
            ValidateToolBehaviour(
                tool_fn_name="invoke_papers_agent", tool_call_wanted=False
            ),
        ],
    ),
    Case(
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
    # Case(
    #     inputs="Check what our most recently saved papers are (we don't save papers on weekends), and rank them for me.",
    #     evaluators=[
    #         HasMatchingSpan(
    #             query={"attributes": {"gen_ai.tool.name": "invoke_papers_agent"}},
    #         ),
    #     ],
    # ),
]


@pytest.fixture(scope="module", autouse=True)
async def cleanup_database():
    """Runs once before all tests in this file."""
    # Run your nuking logic here
    await dev_nuke_redis()
    DEV_DB.execute(
        f"DELETE FROM {DEV_DB.rankings_table} WHERE user_id = %s", (TEST_USER_ID,)
    )
    DEV_DB.execute(
        f"DELETE FROM {DEV_DB.facts_table} WHERE user_id = %s", (TEST_USER_ID,)
    )
    yield  # Tests come through here


@pytest.mark.asyncio
@pytest.mark.parametrize("case", JOURNEY_CASES)
async def test_user_journey(case: Case):
    # This acts as the 'driver' for the individual snapshot
    async def run_snapshot(inp):
        # Your actual app logic
        res = await handle_query(
            MasterRequest(user_id=TEST_USER_ID, session_id=TEST_SESSION_ID, query=inp),
            db=DEV_DB,
        )
        return res

    # Wrap the single case in a Dataset to use the evaluator engine
    dataset = Dataset(cases=[case])
    report = await dataset.evaluate(run_snapshot)
    print(report)
    # assert len(report.failures) == 0, f"Failed on {report.failures}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
