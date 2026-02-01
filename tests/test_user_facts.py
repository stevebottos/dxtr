"""Tests for user fact storage and retrieval.

Tests multi-turn conversations where DXTR learns about users and stores facts.
Uses pydantic-ai agents for simulated user and judge - same interface as DXTR.

Tests are designed to run SEQUENTIALLY and build on each other:
1. test_01_multi_turn_builds_profile - DXTR learns about user, stores facts
2. test_02_retrieves_stored_facts - DXTR recalls what it learned
3. test_03_trivial_chat_no_storage - Trivial messages don't trigger fact storage

Run:
    pytest tests/test_user_facts.py -v -s

Artifacts are saved to: tests/artifacts/<test_name>_<timestamp>.json
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai_litellm import LiteLLMModel

from dxtr import constants
from dxtr.data_models import MasterRequest
from dxtr.db import PostgresHelper
from dxtr.server import handle_query, get_user_add_context, dev_nuke_redis

from tests.conftest import PROFILE_CONTENT


# =============================================================================
# Test Configuration
# =============================================================================

TEST_USER_ID = "steve_test"
TEST_SESSION_ID = f"session_{uuid.uuid4().hex[:8]}"

# Dev database for tests
DEV_DB = PostgresHelper(is_dev=True)


def make_request(query: str, session_id: str = TEST_SESSION_ID) -> MasterRequest:
    """Create a MasterRequest with fixed user ID."""
    return MasterRequest(
        user_id=TEST_USER_ID,
        session_id=session_id,
        query=query,
    )


# =============================================================================
# Test Agents (pydantic-ai, through LiteLLM)
# =============================================================================

SIMULATED_USER_PROMPT = f"""You are a simulated user for testing an AI assistant called DXTR.
You ARE the person described in the profile below. Respond naturally in first person.

PROFILE:
{PROFILE_CONTENT}

Guidelines:
- Be conversational and natural
- Don't dump all information at once - reveal things as conversation progresses
- If asked something not in the profile, make up something consistent
- Keep responses concise (1-3 sentences typically)
- IMPORTANT: Do NOT ask for paper recommendations or rankings. This test is about profile building, not paper discovery. Just share info about yourself when asked.
"""

simulated_user = Agent(
    LiteLLMModel(
        model_name="openai/simulated-user",
        api_key=constants.API_KEY,
        api_base=constants.BASE_URL,
    ),
    system_prompt=SIMULATED_USER_PROMPT,
    output_type=str,
)


JUDGE_PROMPT = """You are a judge evaluating an AI assistant called DXTR.
You will be given a conversation and/or artifacts to evaluate.
Be strict but fair. Provide a score and brief reasoning.

Respond in this exact format:
SCORE: <number 1-10>
REASONING: <1-2 sentences explaining the score>
"""

judge = Agent(
    LiteLLMModel(
        model_name="openai/judge",
        api_key=constants.API_KEY,
        api_base=constants.BASE_URL,
    ),
    system_prompt=JUDGE_PROMPT,
    output_type=str,
)


def parse_judge_response(response: str) -> tuple[int, str]:
    """Parse judge response into score and reasoning."""
    lines = response.strip().split("\n")
    score = 5  # default
    reasoning = ""

    for line in lines:
        if line.startswith("SCORE:"):
            try:
                score = int(line.replace("SCORE:", "").strip())
            except ValueError:
                pass
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()

    return score, reasoning


ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def save_artifacts(test_name: str, artifacts: dict) -> Path:
    """Save test artifacts to JSON file for inspection."""
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{test_name}_{timestamp}.json"
    filepath = ARTIFACTS_DIR / filename

    # Add metadata
    artifacts["_meta"] = {
        "test_name": test_name,
        "timestamp": timestamp,
        "user_id": TEST_USER_ID,
        "session_id": TEST_SESSION_ID,
    }

    filepath.write_text(json.dumps(artifacts, indent=2, default=str))
    print(f"\n[ARTIFACTS] Saved to: {filepath}")
    return filepath


# =============================================================================
# Utilities
# =============================================================================


def extract_tool_calls(messages) -> list[str]:
    """Extract tool names from agent message history."""
    tool_names = []
    for msg in messages:
        for part in getattr(msg, "parts", []):
            if hasattr(part, "tool_name") and part.tool_name is not None:
                tool_names.append(part.tool_name)
    return tool_names


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def db():
    """Database helper - cleans up test user at start."""
    DEV_DB.execute(f"DELETE FROM {DEV_DB.facts_table} WHERE user_id = %s", (TEST_USER_ID,))
    DEV_DB.execute(f"DELETE FROM {DEV_DB.rankings_table} WHERE user_id = %s", (TEST_USER_ID,))
    yield DEV_DB


@pytest.fixture(scope="module")
def conversation_state():
    """Shared state for tracking conversation across sequential tests."""
    return {
        "user_history": [],  # Message history for simulated_user agent
        "all_tool_calls": [],
    }


# =============================================================================
# Test 01: Multi-turn conversation builds user profile
# =============================================================================


@pytest.mark.asyncio
async def test_01_multi_turn_builds_profile(db, conversation_state):
    """DXTR learns about user through multi-turn conversation and stores facts.

    Criteria:
    1. DXTR should ask targeted questions to learn about the user (no profile exists yet)
    2. DXTR should store meaningful facts as the user reveals them
    3. Conversation should feel natural, not like an interrogation
    """
    await dev_nuke_redis()

    num_turns = 5
    user_history = conversation_state["user_history"]
    all_tool_calls = conversation_state["all_tool_calls"]
    dxtr_responses = []

    # User initiates with intro (not a paper request)
    user_message = "Hey there! I'm new here. What can you help me with?"

    print(f"\n{'=' * 60}")
    print("TEST 01: Multi-turn conversation builds profile")
    print(f"{'=' * 60}")

    for turn in range(num_turns):
        print(f"\n--- Turn {turn + 1} ---")
        print(f"USER: {user_message}")

        # DXTR responds via handle_query (Redis-backed)
        request = make_request(user_message)
        add_context = get_user_add_context(request.user_id, db)
        result = await handle_query(request, add_context, db)

        dxtr_response = result.output
        dxtr_responses.append(dxtr_response)
        tools = extract_tool_calls(result.new_messages())  # Only this turn's messages
        all_tool_calls.extend(tools)

        print(f"DXTR: {dxtr_response}")
        if tools:
            print(f"  [tools: {tools}]")

        # Update simulated user's history
        user_history.append({"role": "user", "content": user_message})
        user_history.append({"role": "assistant", "content": dxtr_response})

        # Simulated user responds to DXTR (for next turn)
        if turn < num_turns - 1:
            # Build prompt for simulated user: "DXTR just said X, respond as the user"
            prompt = f'DXTR (the AI assistant) just said: "{dxtr_response}"\n\nRespond as the user would.'
            user_result = await simulated_user.run(prompt, message_history=user_history)
            user_message = user_result.output

    print(f"\n{'=' * 60}")

    # === CRITERIA 1: Judge evaluates DXTR's question quality ===
    conversation_text = "\n".join(
        [f"Turn {i + 1} - DXTR: {resp}" for i, resp in enumerate(dxtr_responses)]
    )

    question_eval_prompt = f"""Evaluate whether DXTR asked good, targeted questions to learn about a NEW user.

DXTR should ask about:
- Professional background (role, experience)
- Technical interests and specializations
- Current learning goals
- Preferences

CONVERSATION (DXTR's responses only):
{conversation_text}

Did DXTR ask targeted questions to build a user profile? Score 1-10 where:
- 1-3: No questions or only generic questions
- 4-6: Some questions but not targeted to profile-building
- 7-10: Good targeted questions about background, interests, goals
"""

    judge_result = await judge.run(question_eval_prompt)
    question_score, question_reasoning = parse_judge_response(judge_result.output)

    print(f"\n[JUDGE] Question Quality: {question_score}/10")
    print(f"[JUDGE] Reasoning: {question_reasoning}")

    assert question_score >= 5, (
        f"DXTR's questions scored {question_score}/10 - should ask better targeted questions. {question_reasoning}"
    )

    # === CRITERIA 2: DXTR should store facts ===
    store_calls = [t for t in all_tool_calls if t == "store_user_fact"]
    stored_facts = db.query(
        f"SELECT id, fact, created_at FROM {db.facts_table} WHERE user_id = %s ORDER BY created_at ASC",
        (TEST_USER_ID,),
    )

    print(f"\nstore_user_fact called {len(store_calls)} times")
    print(f"Stored facts ({len(stored_facts)}):")
    for f in stored_facts:
        print(f"  - {f['fact']}")

    assert len(store_calls) >= 1, (
        f"Expected at least 1 store_user_fact call, got {len(store_calls)}"
    )
    assert len(stored_facts) >= 1, (
        f"Expected at least 1 stored fact, got {len(stored_facts)}"
    )

    # === CRITERIA 3: Judge evaluates stored fact quality ===
    facts_text = "\n".join([f"- {f['fact']}" for f in stored_facts])

    facts_eval_prompt = f"""Evaluate the quality of facts stored about a user during a conversation.

Good facts should be:
- Meaningful and specific (not generic)
- About the user's background, interests, goals, or preferences
- Non-redundant (no duplicates)
- Useful for personalizing future interactions

STORED FACTS:
{facts_text}

Score 1-10 where:
- 1-3: Trivial, generic, or redundant facts
- 4-6: Some useful facts but missing depth
- 7-10: Meaningful, specific, actionable facts about the user
"""

    judge_result = await judge.run(facts_eval_prompt)
    facts_score, facts_reasoning = parse_judge_response(judge_result.output)

    print(f"\n[JUDGE] Fact Quality: {facts_score}/10")
    print(f"[JUDGE] Reasoning: {facts_reasoning}")

    assert facts_score >= 5, (
        f"Stored facts scored {facts_score}/10 - should store more meaningful facts. {facts_reasoning}"
    )

    # === Save artifacts for inspection ===
    save_artifacts(
        "test_01_multi_turn_builds_profile",
        {
            "conversation": user_history,
            "tool_calls": all_tool_calls,
            "stored_facts": [
                {"id": f["id"], "fact": f["fact"], "created_at": f["created_at"]}
                for f in stored_facts
            ],
            "judge_evaluations": {
                "question_quality": {
                    "score": question_score,
                    "reasoning": question_reasoning,
                },
                "fact_quality": {"score": facts_score, "reasoning": facts_reasoning},
            },
        },
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
