"""DeepEval test configuration and shared fixtures."""

import os
import pytest
import uuid
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from deepeval.test_case import LLMTestCase, ToolCall

from dxtr import set_session_id, set_session_tags, set_session_metadata, set_session_state, get_model_settings
from dxtr.agents.util import load_session_state
from dxtr.agents.master import agent
from dxtr.data_models import MasterRequest, SessionState

from tests.fixtures import papers as paper_fixtures


# =============================================================================
# Test User Configuration
# =============================================================================

# Single user ID for the entire test session (simulates real user journey)
TEST_USER_ID = f"deepeval_test_{uuid.uuid4().hex[:8]}"

# Profile content for profile creation tests
PROFILE_CONTENT = (Path(__file__).parent.parent / "testing_strat" / "profile.md").read_text()


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def test_user_id():
    """Shared user ID across all tests in the session."""
    return TEST_USER_ID


@pytest.fixture
def profile_content():
    """Profile content for profile creation tests."""
    return PROFILE_CONTENT


@pytest.fixture
def make_request(test_user_id):
    """Factory fixture to create MasterRequest with consistent user ID."""
    def _make_request(query: str, session_id: str | None = None) -> MasterRequest:
        return MasterRequest(
            user_id=test_user_id,
            session_id=session_id or f"session-{uuid.uuid4().hex[:8]}",
            query=query,
        )
    return _make_request


@pytest.fixture
async def setup_session(test_user_id):
    """Setup session state and tracing for a test."""
    async def _setup(phase: str, state: SessionState | None = None):
        set_session_id(f"{phase}-{test_user_id}")
        set_session_tags(["test", "deepeval"])
        set_session_metadata({"phase": phase, "test_user_id": test_user_id})

        if state:
            set_session_state(state)
        else:
            # Load actual state from GCS
            loaded_state = await load_session_state(test_user_id)
            set_session_state(loaded_state)

        return get_model_settings()

    return _setup


# =============================================================================
# Database Mocking
# =============================================================================

class MockPostgresHelper:
    """Mock database that uses fixture data with dynamic dates.

    Papers are assigned dates relative to "today" at test time:
    - First 8 papers -> today
    - Next 8 papers -> yesterday

    This allows testing date-aware queries without a real database.
    """

    def get_available_dates(self, days_back: int = 7) -> list[dict]:
        return paper_fixtures.get_available_dates(days_back)

    def get_papers_by_date(self, target_date) -> list[dict]:
        if isinstance(target_date, str):
            target_date = date.fromisoformat(target_date)
        return paper_fixtures.get_papers_for_date(target_date)

    def get_top_papers(self, target_date, limit: int = 10) -> list[dict]:
        if isinstance(target_date, str):
            target_date = date.fromisoformat(target_date)
        return paper_fixtures.get_top_papers(target_date, limit)

    def get_papers_for_ranking(self, target_date) -> dict[str, dict]:
        if isinstance(target_date, str):
            target_date = date.fromisoformat(target_date)
        return paper_fixtures.get_papers_for_ranking(target_date)

    def get_paper_stats(self, days_back: int = 7) -> dict:
        return paper_fixtures.get_paper_stats(days_back)

    def get_paper_count(self, days_back: int | None = None) -> int:
        return 16

    def get_date_with_most_papers(self, days_back: int = 7) -> dict | None:
        today = date.today()
        return {"date": today.isoformat(), "count": 8}


@pytest.fixture(autouse=True)
def mock_database():
    """Automatically mock PostgresHelper for all tests."""
    with patch("dxtr.db.PostgresHelper", MockPostgresHelper):
        with patch("dxtr.agents.master.PostgresHelper", MockPostgresHelper):
            yield


# =============================================================================
# Helper Functions
# =============================================================================

def extract_tool_calls(messages) -> list[ToolCall]:
    """Extract tool calls from agent message history as DeepEval ToolCall objects."""
    tool_calls = []
    for msg in messages:
        for part in getattr(msg, "parts", []):
            if hasattr(part, "tool_name"):
                tool_calls.append(ToolCall(name=part.tool_name))
    return tool_calls


async def run_agent_and_build_test_case(
    query: str,
    request: MasterRequest,
    expected_tools: list[str],
    model_settings: dict | None = None,
    message_history: list | None = None,
) -> LLMTestCase:
    """Run the agent and build a DeepEval LLMTestCase.

    Args:
        query: User input
        request: MasterRequest with user/session info
        expected_tools: List of tool names expected to be called
        model_settings: Optional model settings
        message_history: Optional conversation history

    Returns:
        LLMTestCase ready for DeepEval metrics
    """
    result = await agent.run(
        query,
        deps=request,
        model_settings=model_settings or {},
        message_history=message_history or [],
    )

    tools_called = extract_tool_calls(result.all_messages())

    return LLMTestCase(
        input=query,
        actual_output=result.output,
        tools_called=tools_called,
        expected_tools=[ToolCall(name=t) for t in expected_tools],
    )
