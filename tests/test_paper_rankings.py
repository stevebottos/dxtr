"""Tests for papers ranking agent.

Tests:
1. Tool invocation - master agent correctly calls papers agent when appropriate
2. Multi-turn conversation - different ranking methods and follow-up questions

Run:
    pytest tests/test_paper_rankings.py -v -s
"""

import pytest
from datetime import date

from pydantic_ai.messages import ToolCallPart

from dxtr.data_models import MasterRequest, AddContext
from dxtr.server import handle_query, dev_nuke_redis


# === Test Configuration ===

TEST_USER_ID = "papers_test_user"
TEST_SESSION_ID = "papers_test_session"
TODAY = date.today().isoformat()

SAMPLE_PROFILE = """Known facts about user (4 total):
- User is a machine learning engineer with 7 years experience
- User is interested in multimodal LLMs and agentic systems
- User has limited compute (12GB VRAM) and prefers efficient architectures
- User is an applied AI scientist focused on using foundation models in systems, more engineering-focused than research"""

# Today's papers fixture (sorted by upvotes)
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


# === Helpers ===


def make_request(query: str, session_id: str = TEST_SESSION_ID) -> MasterRequest:
    return MasterRequest(
        user_id=TEST_USER_ID,
        session_id=session_id,
        query=query,
    )


def make_context(with_papers: bool = False) -> AddContext:
    return AddContext(
        user_profile_facts=SAMPLE_PROFILE,
        today_date=TODAY,
        papers_by_date={TODAY: TODAYS_PAPERS} if with_papers else None,
    )


def extract_tool_calls(messages) -> list[str]:
    """Extract tool names from agent message history."""
    return [
        part.tool_name
        for msg in messages
        for part in getattr(msg, "parts", [])
        if isinstance(part, ToolCallPart)
    ]


def normalize_text(text: str) -> str:
    """Normalize unicode characters for comparison."""
    return text.replace("‑", "-").replace("–", "-").replace("—", "-")


def response_contains(response: str, text: str) -> bool:
    """Check if response contains text (case-insensitive, unicode-normalized)."""
    return normalize_text(text.lower()) in normalize_text(response.lower())


# === Test 1: Tool Invocation ===

INVOCATION_TEST_CASES = [
    # Should invoke papers rank agent
    ("Rank today's papers for me", True),
    ("What papers should I read today?", True),
    ("Show me paper recommendations", True),
    ("Rank papers from 2024-01-15", True),
    ("What are the best papers for me?", True),
    # Should NOT invoke papers rank agent
    ("Hello, how are you?", False),
    ("What's my profile?", False),
    ("I'm interested in transformers", False),
    ("Tell me about yourself", False),
    ("What can you help me with?", False),
]


@pytest.mark.asyncio
async def test_tool_invocation():
    """Test that master agent correctly invokes papers rank agent when appropriate."""

    context = make_context()

    for query, should_invoke in INVOCATION_TEST_CASES:
        await dev_nuke_redis()

        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print(f"Expected invoke_papers_rank_agent: {should_invoke}")

        request = make_request(query)
        result = await handle_query(request, context)

        tool_calls = extract_tool_calls(result.new_messages())
        did_invoke = "invoke_papers_rank_agent" in tool_calls

        print(f"Tool calls: {tool_calls}")
        print(f"Did invoke: {did_invoke}")
        print(f"Response: {result.output[:100]}...")

        if should_invoke:
            assert did_invoke, (
                f"Expected invoke_papers_rank_agent for query '{query}', "
                f"but got tool calls: {tool_calls}"
            )
        else:
            assert not did_invoke, (
                f"Did NOT expect invoke_papers_rank_agent for query '{query}', "
                f"but it was called. Tool calls: {tool_calls}"
            )

        await dev_nuke_redis()

    print(f"\n{'=' * 60}")
    print(f"All {len(INVOCATION_TEST_CASES)} invocation test cases passed!")


# === Test 2: Multi-turn Conversation ===


@pytest.mark.asyncio
async def test_multi_turn_conversation():
    """Test multi-turn conversation with different ranking methods and follow-ups."""

    await dev_nuke_redis()
    context = make_context(with_papers=True)

    print(f"\n{'=' * 60}")
    print("MULTI-TURN PAPER RANKING TEST")
    print(f"{'=' * 60}")

    # === Turn 1: Rank by upvotes ===
    print("\n--- Turn 1: Rank by upvotes ---")
    request = make_request("Show me today's papers ranked by upvotes")
    result = await handle_query(request, context)

    tool_calls = extract_tool_calls(result.new_messages())
    print(f"Tool calls: {tool_calls}")
    print(f"Response preview: {result.output[:200]}...")

    assert "invoke_papers_rank_agent" in tool_calls, "Should invoke papers agent"
    assert response_contains(result.output, "Harder Is Better") or response_contains(
        result.output, "2601.20614"
    ), "Top paper should be in response"

    # === Turn 2: Rank by profile ===
    print("\n--- Turn 2: Rank by profile ---")
    request = make_request("Now rank them based on my interests")
    result = await handle_query(request, context)

    tool_calls = extract_tool_calls(result.new_messages())
    print(f"Tool calls: {tool_calls}")
    print(f"Response preview: {result.output[:200]}...")

    assert "invoke_papers_rank_agent" in tool_calls, "Should invoke papers agent"

    # === Turn 3: Rank by different criteria (specific request) ===
    print("\n--- Turn 3: Rank for formal verification topic ---")
    request = make_request(
        "I need papers specifically about formal verification and mathematical foundations - can you rank today's papers for that topic?"
    )
    result = await handle_query(request, context)

    tool_calls = extract_tool_calls(result.new_messages())
    print(f"Tool calls: {tool_calls}")
    print(f"Response preview: {result.output[:200]}...")

    assert "invoke_papers_rank_agent" in tool_calls, "Should invoke papers agent"

    # === Turn 4: Follow-up about upvotes (NO tool call) ===
    print("\n--- Turn 4: Follow-up about upvotes ---")
    request = make_request("What was the third most upvoted paper?")
    result = await handle_query(request, context)

    tool_calls = extract_tool_calls(result.new_messages())
    print(f"Tool calls: {tool_calls}")
    print(f"Response: {result.output}")

    assert (
        "invoke_papers_rank_agent" not in tool_calls
    ), "Should NOT re-invoke papers agent for follow-up"
    assert response_contains(result.output, "Innovator-VL") or response_contains(
        result.output, "2601.19325"
    ), "Should mention the third most upvoted paper (Innovator-VL)"

    # === Turn 5: Follow-up about profile ranking (NO tool call) ===
    print("\n--- Turn 5: Follow-up about profile ranking ---")
    request = make_request(
        "In the ranking based on my interests, which paper was ranked highest?"
    )
    result = await handle_query(request, context)

    tool_calls = extract_tool_calls(result.new_messages())
    print(f"Tool calls: {tool_calls}")
    print(f"Response: {result.output}")

    assert (
        "invoke_papers_rank_agent" not in tool_calls
    ), "Should NOT re-invoke papers agent for follow-up"

    # === Turn 6: Follow-up about formal verification ranking (NO tool call) ===
    print("\n--- Turn 6: Follow-up about formal verification ranking ---")
    request = make_request(
        "What about in the formal verification ranking - what was number one?"
    )
    result = await handle_query(request, context)

    tool_calls = extract_tool_calls(result.new_messages())
    print(f"Tool calls: {tool_calls}")
    print(f"Response: {result.output}")

    assert (
        "invoke_papers_rank_agent" not in tool_calls
    ), "Should NOT re-invoke papers agent for follow-up"

    print(f"\n{'=' * 60}")
    print("All turns completed successfully!")
    print(f"{'=' * 60}")

    await dev_nuke_redis()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
