"""Reproduce and inspect thinking/reasoning leakage from the model.

Run: pytest tests/test_thinking_leakage.py -v -s
"""

import pytest
import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai_litellm import LiteLLMModel
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

from dxtr import constants
from dxtr.data_models import MasterRequest
from dxtr.server import handle_query
from tests.mocks import InMemoryDB, InMemoryConversationStore

logfire.configure(send_to_logfire=True)
logfire.instrument_pydantic_ai()

DB = InMemoryDB()
STORE = InMemoryConversationStore()
USER_ID = "dev_user_thinking_test"
SESSION_ID = "thinking_test_session"


class JudgeResult(BaseModel):
    passed: bool = Field(
        description="True if the response reads like a clean user-facing message, False if it contains leaked reasoning"
    )
    reasoning: str = Field(
        description="Brief explanation of the verdict"
    )


JUDGE = Agent(
    LiteLLMModel(
        model_name="openai/judge",
        api_key=constants.API_KEY,
        api_base=constants.BASE_URL,
    ),
    output_type=JudgeResult,
    system_prompt="You are a judge evaluating whether an AI assistant's response is clean and user-facing, or whether it contains leaked internal reasoning/thinking.",
    retries=5,
)

JUDGE_PROMPT = """\
The following is a response from an AI research assistant called DXTR.

Evaluate whether this response is a clean, user-facing message — or whether it contains
leaked internal reasoning, chain-of-thought, or "thinking out loud" that was clearly not
meant to be shown to the user.

Signs of leaked thinking:
- The response starts with self-talk like "Okay, the user is asking..." or "Let me think about..."
- It contains meta-reasoning about what tools to use or not use
- It narrates its own decision-making process before giving the actual answer
- The actual user-facing answer appears buried after paragraphs of internal reasoning

A clean response may still be conversational or ask clarifying questions — that's fine.
The key question is: does it contain text that reads like internal monologue rather than
a response directed at the user?

DXTR's response:
---
{response}
---
"""


async def _run_turn(query: str) -> str:
    """Run a single turn and return the output, printing full message details."""
    request = MasterRequest(
        user_id=USER_ID,
        session_id=SESSION_ID,
        query=query,
    )
    result = await handle_query(request, DB, store=STORE)

    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}")

    for i, msg in enumerate(result.new_messages()):
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    print(f"  [TextPart] ({len(part.content)} chars):")
                    print(f"    {part.content}")
                elif isinstance(part, ToolCallPart):
                    print(f"  [ToolCallPart] {part.tool_name}({part.args})")

    print(f"\n--- Final output ({len(result.output)} chars) ---")
    print(result.output)
    print(f"{'='*80}\n")

    return result.output


GREETING_QUERIES = [
    "what do you know about me?",
    "hey! who are you and what can you do?",
    "hi there, I'm new here",
    "can you help me find interesting papers?",
    "hello, what is this?",
    "what kind of papers do you cover?",
    "do you remember anything about my preferences?",
    "how does this work?",
    "yo, what's good?",
    "I just signed up, what should I do first?",
]


@pytest.mark.parametrize("query", GREETING_QUERIES, ids=[q[:30] for q in GREETING_QUERIES])
async def test_no_thinking_leakage(query: str):
    """First message on a fresh session — response should be clean, no leaked reasoning."""
    DB.reset()
    STORE.reset()

    output = await _run_turn(query)

    judge_result = await JUDGE.run(JUDGE_PROMPT.format(response=output))
    print(f"Judge: passed={judge_result.output.passed}, reasoning={judge_result.output.reasoning}")

    assert judge_result.output.passed, (
        f"Thinking leaked into response. Judge reasoning: {judge_result.output.reasoning}"
    )
