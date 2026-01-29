"""
Consolidated E2E test for DXTR user journey.

IMPORTANT: Uses ONE user ID for the ENTIRE test suite.
Tests run sequentially and verify that artifacts (profile) persist across fresh chat sessions.

This simulates a real user who:
1. Creates a profile once
2. Comes back later in new sessions
3. The model retrieves their existing profile (doesn't recreate it)

Run: pytest tests/integration/scenario_test_user_journey.py -v -s
"""

import pytest
from pathlib import Path
from datetime import date, timedelta
import uuid

from dxtr.agents.master import agent
from dxtr.data_models import MasterRequest
from dxtr import set_session_id, set_session_tags, set_session_metadata, get_model_settings


# =============================================================================
# ONE user ID for the entire test suite - generated at module load
# =============================================================================
TEST_USER_ID = f"e2e_test_{uuid.uuid4().hex[:8]}"

TODAY = date.today()
YESTERDAY = TODAY - timedelta(days=1)

# Load profile content from testing_strat/profile.md
PROFILE_CONTENT = (Path(__file__).parent.parent.parent / "testing_strat" / "profile.md").read_text()


def extract_tool_calls(messages) -> list[str]:
    """Extract tool names from agent message history."""
    tool_names = []
    for msg in messages:
        for part in getattr(msg, "parts", []):
            if hasattr(part, "tool_name"):
                tool_names.append(part.tool_name)
    return tool_names


def create_deps(query: str = "") -> MasterRequest:
    """Create deps with the shared test user ID."""
    return MasterRequest(
        user_id=TEST_USER_ID,
        session_id=f"session-{uuid.uuid4().hex[:8]}",  # New session each time (fresh chat)
        query=query,
    )


def setup_tracing(phase: str):
    """Configure Langfuse tracing for test phase."""
    set_session_id(f"{phase}-{TEST_USER_ID}")
    set_session_tags(["test", "integration", "e2e"])
    set_session_metadata({"phase": phase, "test_user_id": TEST_USER_ID})


# =============================================================================
# Test 01: Profile Creation
# =============================================================================

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_01_profile_creation():
    """
    Create profile from provided info.

    Send profile.md content and ask model to create profile.
    Model may either create directly or ask permission first - both acceptable.

    Expected tools: create_github_summary, call_profile_synthesizer
    """
    setup_tracing("profile_creation")

    print(f"\n{'='*60}")
    print(f"TEST 01: Profile Creation")
    print(f"User ID: {TEST_USER_ID}")
    print(f"{'='*60}\n")

    user_message = f"""Here's all my information for creating my profile. Please create it based on this:

{PROFILE_CONTENT}

That's everything you need - go ahead and create my profile."""

    deps = create_deps(user_message)

    print(f"Sending profile info to model...")
    result = await agent.run(
        user_message,
        deps=deps,
        model_settings=get_model_settings(),
    )

    response = result.output
    messages = result.all_messages()
    tools_called = extract_tool_calls(messages)

    print(f"\nTools called: {tools_called}")
    print(f"Response preview: {response[:300]}...")

    profile_created = "call_profile_synthesizer" in tools_called
    github_summarized = "create_github_summary" in tools_called
    asked_permission = not profile_created and any(
        word in response.lower() for word in ["permission", "confirm", "ready", "shall i", "would you like"]
    )

    if profile_created:
        print(f"\nModel created profile directly.")
        assert github_summarized, "Expected create_github_summary to be called with GitHub links"
    elif asked_permission:
        print(f"\nModel asked for permission. Sending confirmation...")

        confirm_message = "Yes, please create my profile."
        result2 = await agent.run(
            confirm_message,
            deps=create_deps(confirm_message),
            message_history=messages,
            model_settings=get_model_settings(),
        )

        response = result2.output
        all_messages = result2.all_messages()
        tools_called = extract_tool_calls(all_messages)

        print(f"Tools called after confirmation: {tools_called}")
        assert "call_profile_synthesizer" in tools_called, f"Expected profile to be created. Tools: {tools_called}"
    else:
        pytest.fail(f"Unexpected behavior. Tools: {tools_called}, Response: {response[:500]}")

    print(f"\nPASSED: Profile created for {TEST_USER_ID}")


# =============================================================================
# Test 02: Top Papers by Upvotes (Fresh Chat)
# =============================================================================

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_02_top_papers_by_upvotes_yesterday():
    """
    Fresh chat - get top papers by upvotes from yesterday.

    This is NOT personalized ranking, just upvote-based.
    Profile exists from test_01 but shouldn't be needed here.

    Expected tools: get_top_papers
    """
    setup_tracing("top_papers_upvotes")

    print(f"\n{'='*60}")
    print(f"TEST 02: Top Papers by Upvotes (Yesterday)")
    print(f"User ID: {TEST_USER_ID} (fresh chat, no history)")
    print(f"{'='*60}\n")

    user_message = "Show me the top 5 papers based on upvotes from yesterday."
    deps = create_deps(user_message)

    print(f"Query: {user_message}")
    result = await agent.run(
        user_message,
        deps=deps,
        model_settings=get_model_settings(),
    )

    response = result.output
    tools_called = extract_tool_calls(result.all_messages())

    print(f"\nTools called: {tools_called}")
    print(f"Response preview: {response[:300]}...")

    assert "get_top_papers" in tools_called, f"Expected get_top_papers. Called: {tools_called}"
    # Should NOT try to create profile - it already exists
    assert "call_profile_synthesizer" not in tools_called, "Should not recreate profile"

    print(f"\nPASSED")


# =============================================================================
# Test 03: Rank Papers by Profile - Yesterday (Fresh Chat)
# =============================================================================

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_03_rank_by_profile_yesterday():
    """
    Fresh chat - rank papers by profile from yesterday.

    Model should:
    1. Check profile state (finds existing profile from test_01)
    2. Rank papers using that profile
    3. NOT try to create a new profile

    Expected tools: check_profile_state, rank_papers_for_user
    """
    setup_tracing("rank_profile_yesterday")

    print(f"\n{'='*60}")
    print(f"TEST 03: Rank by Profile (Yesterday)")
    print(f"User ID: {TEST_USER_ID} (fresh chat, no history)")
    print(f"{'='*60}\n")

    user_message = "Rank the top 5 papers for me based on my profile from yesterday."
    deps = create_deps(user_message)

    print(f"Query: {user_message}")
    result = await agent.run(
        user_message,
        deps=deps,
        model_settings=get_model_settings(),
    )

    response = result.output
    tools_called = extract_tool_calls(result.all_messages())

    print(f"\nTools called: {tools_called}")
    print(f"Response preview: {response[:300]}...")

    assert "check_profile_state" in tools_called, f"Should check for existing profile. Tools: {tools_called}"
    assert "rank_papers_for_user" in tools_called, f"Should rank papers. Tools: {tools_called}"
    # Critical: should NOT recreate profile
    assert "call_profile_synthesizer" not in tools_called, "Should NOT recreate profile - it exists!"

    print(f"\nPASSED")


# =============================================================================
# Test 04: Rank Papers by Profile - Today (Fresh Chat)
# =============================================================================

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_04_rank_by_profile_today():
    """
    Fresh chat - rank papers by profile from today.

    Model should retrieve existing profile and rank today's papers.

    Expected tools: check_profile_state, rank_papers_for_user
    """
    setup_tracing("rank_profile_today")

    print(f"\n{'='*60}")
    print(f"TEST 04: Rank by Profile (Today)")
    print(f"User ID: {TEST_USER_ID} (fresh chat, no history)")
    print(f"{'='*60}\n")

    user_message = "Rank the top 5 papers for me based on my profile from today."
    deps = create_deps(user_message)

    print(f"Query: {user_message}")
    result = await agent.run(
        user_message,
        deps=deps,
        model_settings=get_model_settings(),
    )

    response = result.output
    tools_called = extract_tool_calls(result.all_messages())

    print(f"\nTools called: {tools_called}")
    print(f"Response preview: {response[:300]}...")

    assert "rank_papers_for_user" in tools_called, f"Should rank papers. Tools: {tools_called}"
    # Critical: should NOT recreate profile
    assert "call_profile_synthesizer" not in tools_called, "Should NOT recreate profile - it exists!"

    print(f"\nPASSED")


# =============================================================================
# Test 05: Greeting - No Expensive Tools
# =============================================================================

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_05_greeting_no_expensive_tools():
    """
    Fresh chat - simple greeting should not trigger expensive tools.

    Should NOT trigger: rank_papers_for_user, call_profile_synthesizer, create_github_summary
    """
    setup_tracing("greeting")

    print(f"\n{'='*60}")
    print(f"TEST 05: Greeting - No Expensive Tools")
    print(f"User ID: {TEST_USER_ID} (fresh chat)")
    print(f"{'='*60}\n")

    user_message = "Hello!"
    deps = create_deps(user_message)

    print(f"Query: {user_message}")
    result = await agent.run(
        user_message,
        deps=deps,
        model_settings=get_model_settings(),
    )

    response = result.output
    tools_called = extract_tool_calls(result.all_messages())

    print(f"\nTools called: {tools_called}")
    print(f"Response preview: {response[:200]}...")

    expensive_tools = {"rank_papers_for_user", "call_profile_synthesizer", "create_github_summary"}
    for tool in expensive_tools:
        assert tool not in tools_called, f"Expensive tool {tool} should not be called for greeting"

    print(f"\nPASSED")


# =============================================================================
# Test 06: Papers Query - No Profile Recreation
# =============================================================================

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_06_papers_query_no_profile_creation():
    """
    Fresh chat - asking about papers should not trigger profile creation.

    Profile exists, but this query doesn't need it. Should not recreate.

    Should NOT trigger: call_profile_synthesizer, create_github_summary
    """
    setup_tracing("papers_no_profile")

    print(f"\n{'='*60}")
    print(f"TEST 06: Papers Query - No Profile Creation")
    print(f"User ID: {TEST_USER_ID} (fresh chat)")
    print(f"{'='*60}\n")

    user_message = "What are today's ML papers?"
    deps = create_deps(user_message)

    print(f"Query: {user_message}")
    result = await agent.run(
        user_message,
        deps=deps,
        model_settings=get_model_settings(),
    )

    response = result.output
    tools_called = extract_tool_calls(result.all_messages())

    print(f"\nTools called: {tools_called}")
    print(f"Response preview: {response[:200]}...")

    assert "call_profile_synthesizer" not in tools_called, "Should not create profile for papers query"
    assert "create_github_summary" not in tools_called, "Should not summarize github for papers query"

    print(f"\nPASSED")


# =============================================================================
# CLI Runner
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def run_all():
        print("\n" + "="*60)
        print(f"RUNNING ALL E2E TESTS")
        print(f"User ID: {TEST_USER_ID}")
        print("="*60)

        await test_01_profile_creation()
        await test_02_top_papers_by_upvotes_yesterday()
        await test_03_rank_by_profile_yesterday()
        await test_04_rank_by_profile_today()
        await test_05_greeting_no_expensive_tools()
        await test_06_papers_query_no_profile_creation()

        print("\n" + "="*60)
        print(f"ALL E2E TESTS PASSED")
        print(f"User ID: {TEST_USER_ID}")
        print("="*60)

    asyncio.run(run_all())
