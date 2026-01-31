"""Conversational (multi-turn) tests using DeepEval.

Tests that the agent retains knowledge across conversation turns,
specifically for the artifact-based ranking flow.

Run:
    pytest tests/test_conversational.py -v
    pytest tests/test_conversational.py -v -k "followup"
"""

import os
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.test_case import ConversationalTestCase, Turn, ToolCall
from deepeval.metrics import ConversationCompletenessMetric, TurnRelevancyMetric
from deepeval.models import GPTModel

from dxtr import set_session_state, get_model_settings
from dxtr.agents.master import agent
from dxtr.data_models import MasterRequest
from dxtr.storage import SessionState, get_store, set_store, InMemoryStore, get_session_key

from tests.conftest import extract_tool_calls


# =============================================================================
# Fixtures
# =============================================================================

PROFILE_FIXTURE_PATH = Path(__file__).parent.parent / "profile_fixture.md"


@pytest.fixture
def profile_content():
    """Load the profile fixture for testing."""
    return PROFILE_FIXTURE_PATH.read_text()


@pytest.fixture
def session_with_profile(profile_content):
    """Create a session state with profile pre-loaded (bypasses profile creation)."""
    return SessionState(
        has_synthesized_profile=True,
        has_github_summary=True,
        profile_content=profile_content,
    )


@pytest.fixture
def fresh_store():
    """Fresh in-memory store for each test."""
    store = InMemoryStore()
    set_store(store)
    return store


@pytest.fixture
def make_request(test_user_id):
    """Factory to create requests with consistent user ID."""
    def _make(query: str, session_id: str = "conv-test-session"):
        return MasterRequest(
            user_id=test_user_id,
            session_id=session_id,
            query=query,
        )
    return _make


# =============================================================================
# DeepEval Configuration
# =============================================================================

openrouter_model = GPTModel(
    model="deepseek/deepseek-chat-v3-0324",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# ConversationCompletenessMetric: Did the agent satisfy user needs?
# E.g., did it rank papers when asked? Did it correctly identify the third paper?
# Threshold at 0.65 to account for LLM evaluator variance while maintaining quality bar.
completeness_metric = ConversationCompletenessMetric(
    model=openrouter_model,
    threshold=0.65,
    include_reason=True,
)

# TurnRelevancyMetric: Were responses relevant to user queries?
# Threshold at 0.65 to account for LLM evaluator variance while maintaining quality bar.
relevancy_metric = TurnRelevancyMetric(
    model=openrouter_model,
    threshold=0.65,
    include_reason=True,
)


# =============================================================================
# Helper: Run Multi-Turn Conversation
# =============================================================================

async def run_conversation(
    queries: list[str],
    make_request,
    session_with_profile,
    fresh_store,
) -> tuple[list[Turn], list]:
    """
    Run a multi-turn conversation and return DeepEval Turns.

    Args:
        queries: List of user queries in order
        make_request: Request factory fixture
        session_with_profile: Pre-loaded session state
        fresh_store: Clean storage instance

    Returns:
        (turns, all_results) - DeepEval Turn objects and raw results
    """
    turns = []
    results = []
    history = []

    session_key = get_session_key("test_user", "conv-test-session")

    # Save initial state with profile
    await fresh_store.save_state(session_key, session_with_profile)

    for query in queries:
        # Set session state for this turn
        state = await fresh_store.get_state(session_key)
        set_session_state(state)

        request = make_request(query)

        # Run agent
        result = await agent.run(
            query,
            deps=request,
            message_history=history,
            model_settings=get_model_settings(),
        )

        # Extract output (plain string now)
        output: str = result.output
        # Only get tools from THIS turn's new messages, not all history
        tools_called = extract_tool_calls(result.new_messages())

        # Build turns
        turns.append(Turn(role="user", content=query))
        turns.append(Turn(
            role="assistant",
            content=output,
            tools_called=[ToolCall(name=tc.name) for tc in tools_called] if tools_called else None,
        ))

        # Update history for next turn
        history = result.all_messages()
        await fresh_store.save_history(session_key, history)

        results.append(result)

    return turns, results


# =============================================================================
# Conversational Tests: Ranking Follow-ups
# =============================================================================

@pytest.mark.asyncio
async def test_ranking_followup_what_was_third(
    make_request,
    session_with_profile,
    fresh_store,
    test_user_id,
):
    """Test: User asks about a specific paper after ranking."""
    turns, results = await run_conversation(
        queries=[
            "Rank today's papers for me",
            "What was the third paper in those rankings?",
        ],
        make_request=make_request,
        session_with_profile=session_with_profile,
        fresh_store=fresh_store,
    )

    # Debug: Print conversation
    print("\n=== CONVERSATION ===")
    for turn in turns:
        print(f"[{turn.role}]: {turn.content[:200]}...")
        if turn.tools_called:
            print(f"  Tools: {[tc.name for tc in turn.tools_called]}")
    print("=== END ===\n")

    test_case = ConversationalTestCase(
        scenario="User requests paper rankings, then asks about a specific paper",
        expected_outcome="Agent should remember and correctly identify the third ranked paper",
        turns=turns,
    )

    assert_test(test_case, [completeness_metric, relevancy_metric])


@pytest.mark.asyncio
async def test_ranking_followup_why_relevant(
    make_request,
    session_with_profile,
    fresh_store,
    test_user_id,
):
    """Test: User asks why a paper is relevant after ranking."""
    turns, results = await run_conversation(
        queries=[
            "Rank today's papers for me",
            "Why is the top paper relevant to my interests?",
        ],
        make_request=make_request,
        session_with_profile=session_with_profile,
        fresh_store=fresh_store,
    )

    test_case = ConversationalTestCase(
        scenario="User requests rankings, then asks about relevance",
        expected_outcome="Agent should explain why top paper matches user's profile",
        turns=turns,
    )

    assert_test(test_case, [completeness_metric, relevancy_metric])


@pytest.mark.asyncio
async def test_ranking_followup_compare_papers(
    make_request,
    session_with_profile,
    fresh_store,
    test_user_id,
):
    """Test: User asks to compare papers after ranking."""
    turns, results = await run_conversation(
        queries=[
            "Rank today's papers for me",
            "Compare the top two papers for me",
        ],
        make_request=make_request,
        session_with_profile=session_with_profile,
        fresh_store=fresh_store,
    )

    test_case = ConversationalTestCase(
        scenario="User requests rankings, then asks for comparison",
        expected_outcome="Agent should compare the first and second ranked papers",
        turns=turns,
    )

    assert_test(test_case, [completeness_metric, relevancy_metric])


@pytest.mark.asyncio
async def test_multiple_ranking_sessions(
    make_request,
    session_with_profile,
    fresh_store,
    test_user_id,
):
    """Test: User requests multiple rankings and asks about both."""
    turns, results = await run_conversation(
        queries=[
            "Rank today's papers for me",
            "Now show me yesterday's papers ranked",
            "Which day had better papers for me?",
        ],
        make_request=make_request,
        session_with_profile=session_with_profile,
        fresh_store=fresh_store,
    )

    test_case = ConversationalTestCase(
        scenario="User requests rankings for multiple days, then compares",
        expected_outcome="Agent should remember both rankings and compare them",
        turns=turns,
    )

    assert_test(test_case, [completeness_metric, relevancy_metric])


@pytest.mark.asyncio
async def test_ranking_then_tangent_then_back(
    make_request,
    session_with_profile,
    fresh_store,
    test_user_id,
):
    """Test: User asks about rankings, asks a tangent question, then returns."""
    turns, results = await run_conversation(
        queries=[
            "Rank today's papers for me",
            "What makes a paper relevant to my profile?",  # Related tangent
            "Going back to those papers - tell me more about paper number 2",
        ],
        make_request=make_request,
        session_with_profile=session_with_profile,
        fresh_store=fresh_store,
    )

    test_case = ConversationalTestCase(
        scenario="User gets rankings, asks about the ranking methodology, returns to specific paper",
        expected_outcome="Agent should explain relevance criteria and still remember the rankings",
        turns=turns,
    )

    assert_test(test_case, [completeness_metric, relevancy_metric])


# =============================================================================
# CLI Runner
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
