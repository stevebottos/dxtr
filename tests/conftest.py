"""Test configuration and shared fixtures."""

import os
import uuid
from pathlib import Path
from datetime import datetime

import pytest
from dotenv import load_dotenv

# Load env vars for database connection
load_dotenv()

# Use dev tables during tests
os.environ["IS_DEV"] = "1"

# Enable deepeval test tracking
os.environ["DEEPEVAL"] = "1"

from dxtr.data_models import MasterRequest
from dxtr.db import PostgresHelper


# === DeepEval Result Saving ===


def pytest_sessionstart(session):
    """Enable deepeval to save test results to .deepeval folder."""
    from deepeval.test_run import global_test_run_manager

    global_test_run_manager.save_to_disk = True
    global_test_run_manager.create_test_run(
        identifier=f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        file_name="tests",
    )


# Test user ID for the session
TEST_USER_ID = f"test_user_{uuid.uuid4().hex[:8]}"

# Load profile fixture for simulated user persona
PROFILE_PATH = Path(__file__).parent.parent / "tests_old" / "fixtures" / "profile.md"
PROFILE_CONTENT = PROFILE_PATH.read_text() if PROFILE_PATH.exists() else ""


@pytest.fixture(scope="session")
def test_user_id():
    """Shared user ID across all tests in the session."""
    return TEST_USER_ID


@pytest.fixture
def profile_content():
    """Profile content for simulated user."""
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
def db(test_user_id):
    """Provide real database connection using dev tables.

    Cleans up test user's data BEFORE test only (not after).
    This allows inspection of test artifacts after the test run.
    """
    helper = PostgresHelper()

    # Clean up before test only
    helper.delete_user_facts(test_user_id)
    helper.delete_paper_rankings(test_user_id)

    yield helper

    # No cleanup after - allows inspection
