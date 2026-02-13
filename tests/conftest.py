"""Test configuration and shared fixtures."""

import uuid

import pytest
from dotenv import load_dotenv

# Load env vars (still needed for LLM API keys etc.)
load_dotenv()

from dxtr.data_models import MasterRequest
from tests.mocks import InMemoryDB


# Shared in-memory database for all tests
DEV_DB = InMemoryDB()

# Test user ID for the session
TEST_USER_ID = f"test_user_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def test_user_id():
    """Shared user ID across all tests in the session."""
    return TEST_USER_ID


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
def db(test_user_id):
    """Provide in-memory database helper.

    Cleans up test user's data BEFORE test only (not after).
    This allows inspection of test artifacts after the test run.
    """
    DEV_DB.execute(
        f"DELETE FROM {DEV_DB.facts_table} WHERE user_id = %s", (test_user_id,)
    )
    DEV_DB.execute(
        f"DELETE FROM {DEV_DB.rankings_table} WHERE user_id = %s", (test_user_id,)
    )

    yield DEV_DB
