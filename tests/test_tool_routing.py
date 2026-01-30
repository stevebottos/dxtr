"""Tool routing tests using DeepEval's ToolCorrectnessMetric.

Tests that the agent calls the right tools for different query types.

Run:
    pytest tests/test_tool_routing.py -v
    pytest tests/test_tool_routing.py -v -k "greeting"  # Run subset
"""

import os
import re
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.models import GPTModel

from dxtr.agents.master import agent
from dxtr.data_models import SessionState

from tests.conftest import extract_tool_calls, PROFILE_CONTENT
from tests.datasets.tool_routing import (
    ToolRoutingCase,
    OutputFormatCase,
    PROFILE_CREATION_CASES,
    PAPER_QUERY_CASES,
    RANKING_CASES,
    NO_TOOL_CASES,
    RANKING_OUTPUT_CASES,
)


# =============================================================================
# DeepEval Model Configuration (OpenRouter)
# =============================================================================

openrouter_model = GPTModel(
    model="openai/gpt-4o-mini",  # Cheap model for eval judging
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


# =============================================================================
# Metric Configuration
# =============================================================================

# Tool correctness metric - checks if expected tools were called
tool_metric = ToolCorrectnessMetric(
    model=openrouter_model,
    threshold=0.5,  # At least 50% of expected tools must be called
    include_reason=True,
)


# =============================================================================
# Test Helpers
# =============================================================================

async def run_test_case(
    case: ToolRoutingCase,
    make_request,
    setup_session,
    profile_content: str | None = None,
    inject_state: SessionState | None = None,
) -> LLMTestCase:
    """Run a tool routing test case and return DeepEval LLMTestCase."""

    # Setup session with optional injected state
    model_settings = await setup_session(case.name, state=inject_state)

    # Replace placeholder with actual profile content
    query = case.input
    if query == "__PROFILE_CONTENT__" and profile_content:
        query = f"""Here's all my information for creating my profile:

{profile_content}

That's everything - please create my profile."""

    request = make_request(query)

    # Run agent
    result = await agent.run(
        query,
        deps=request,
        model_settings=model_settings,
    )

    # Build test case
    tools_called = extract_tool_calls(result.all_messages())

    return LLMTestCase(
        input=query,
        actual_output=result.output,
        tools_called=tools_called,
        expected_tools=[ToolCall(name=t) for t in case.expected_tools],
    )


def assert_forbidden_tools_not_called(test_case: LLMTestCase, forbidden: list[str]):
    """Assert that forbidden tools were not called."""
    called_names = {tc.name for tc in test_case.tools_called}
    for tool in forbidden:
        assert tool not in called_names, f"Forbidden tool '{tool}' was called"


# =============================================================================
# Greeting/Chitchat Tests (No Tools Expected)
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("case", NO_TOOL_CASES, ids=lambda c: c.name)
async def test_no_tools_for_chitchat(case: ToolRoutingCase, make_request, setup_session):
    """Greetings and chitchat should not trigger expensive tools."""
    test_case = await run_test_case(case, make_request, setup_session)

    # For no-tool cases, check forbidden tools aren't called
    assert_forbidden_tools_not_called(test_case, case.forbidden_tools)

    # Also verify no tools at all (or only cheap ones like get_today)
    expensive_tools = {"rank_papers_for_user", "call_profile_synthesizer", "create_github_summary"}
    called_names = {tc.name for tc in test_case.tools_called}
    expensive_called = called_names & expensive_tools
    assert not expensive_called, f"Expensive tools called for chitchat: {expensive_called}"


# =============================================================================
# Paper Query Tests (No Profile Needed)
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("case", PAPER_QUERY_CASES, ids=lambda c: c.name)
async def test_paper_queries(case: ToolRoutingCase, make_request, setup_session):
    """Paper queries should use appropriate paper tools."""
    test_case = await run_test_case(case, make_request, setup_session)

    # Check expected tools with DeepEval
    assert_test(test_case, [tool_metric])

    # Check forbidden tools
    assert_forbidden_tools_not_called(test_case, case.forbidden_tools)


# =============================================================================
# Profile Creation Test (Sequential - Must Run First)
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.order(1)  # Run first in the session
async def test_profile_creation(make_request, setup_session, profile_content):
    """Create profile from provided info - runs first to set up state for later tests."""
    case = PROFILE_CREATION_CASES[0]
    test_case = await run_test_case(
        case,
        make_request,
        setup_session,
        profile_content=profile_content,
    )

    # Verify profile creation tools were called
    assert_test(test_case, [tool_metric])

    called_names = {tc.name for tc in test_case.tools_called}
    assert "call_profile_synthesizer" in called_names, "Profile should be created"


# =============================================================================
# Profile-Based Ranking Tests (Require Existing Profile)
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.order(2)  # Run after profile creation
@pytest.mark.parametrize("case", RANKING_CASES, ids=lambda c: c.name)
async def test_ranking_with_profile(case: ToolRoutingCase, make_request, setup_session):
    """Ranking queries with existing profile should use rank_papers_for_user."""
    # Inject state indicating profile exists (simulates real user with profile)
    injected_state = SessionState(
        has_synthesized_profile=True,
        has_github_summary=True,
        profile_content="Mock profile for testing - user interested in ML, NLP, transformers.",
    )

    test_case = await run_test_case(
        case,
        make_request,
        setup_session,
        inject_state=injected_state,
    )

    # Check expected tools with DeepEval
    assert_test(test_case, [tool_metric])

    # Critical: should NOT recreate profile
    assert_forbidden_tools_not_called(test_case, case.forbidden_tools)


# =============================================================================
# Output Format Tests (Unit Tests)
# =============================================================================

from dxtr.agents.util import format_ranking_results


class TestRankingOutputFormat:
    """Unit tests for ranking output format.

    These tests verify that format_ranking_results includes proper links.
    Expected to FAIL until the formatter is updated to include links.
    """

    def test_ranking_includes_huggingface_links(self):
        """Ranking output should include HuggingFace paper links."""
        # Sample ranking results (same structure as papers_ranking returns)
        results = [
            {"id": "2601.19280", "title": "Great ML Paper", "score": 10, "reason": "Great paper"},
            {"id": "2601.20614", "title": "Good NLP Paper", "score": 8, "reason": "Good paper"},
        ]

        output = format_ranking_results(results)

        # Check HuggingFace links are present
        hf_pattern = r"https://huggingface\.co/papers/\d+\.\d+"
        matches = re.findall(hf_pattern, output)
        assert len(matches) >= 2, f"Expected HuggingFace links for each paper. Output:\n{output}"

    def test_ranking_includes_arxiv_links(self):
        """Ranking output should include arXiv links."""
        results = [
            {"id": "2601.19280", "title": "Great ML Paper", "score": 10, "reason": "Great paper"},
            {"id": "2601.20614", "title": "Good NLP Paper", "score": 8, "reason": "Good paper"},
        ]

        output = format_ranking_results(results)

        # Check arXiv links are present
        arxiv_pattern = r"https://arxiv\.org/abs/\d+\.\d+"
        matches = re.findall(arxiv_pattern, output)
        assert len(matches) >= 2, f"Expected arXiv links for each paper. Output:\n{output}"

    def test_ranking_output_structure(self):
        """Ranking output should have score, title section, and links."""
        results = [
            {"id": "2601.19280", "title": "Great ML Paper", "score": 10, "reason": "Directly relevant"},
        ]

        output = format_ranking_results(results)

        # Should contain the paper ID somewhere
        assert "2601.19280" in output
        # Should contain the score
        assert "10" in output
        # Should contain links
        assert "huggingface.co" in output
        assert "arxiv.org" in output


# =============================================================================
# CLI Runner
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
