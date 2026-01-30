"""Tests for the message bus system.

Tests the two-bus architecture:
- internal_bus: Agent-to-agent communication (tool status, progress)
- user_bus: Agent-to-user communication (rankings, direct content)
"""

import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
import pytest

from dxtr.bus import (
    setup_buses,
    teardown_buses,
    send_to_user,
    send_internal,
    collect_user_content,
)


class TestBusBasics:
    """Test basic bus functionality."""

    def test_send_to_user_puts_content_on_queue(self):
        """send_to_user should put content on the user queue."""
        internal_q, user_q = setup_buses()

        send_to_user("test rankings")

        assert not user_q.empty()
        content = user_q.get_nowait()
        assert content == "test rankings"
        teardown_buses()

    def test_send_internal_puts_event_on_queue(self):
        """send_internal should put event on the internal queue."""
        internal_q, user_q = setup_buses()

        send_internal("tool", "Processing started")

        assert not internal_q.empty()
        event = internal_q.get_nowait()
        assert event["type"] == "tool"
        assert event["message"] == "Processing started"
        teardown_buses()

    def test_collect_user_content(self):
        """collect_user_content should return all content in order."""
        internal_q, user_q = setup_buses()

        send_to_user("First")
        send_to_user("Second")
        send_to_user("Third")

        content = collect_user_content(user_q)
        assert content == ["First", "Second", "Third"]
        teardown_buses()

    def test_buses_are_independent(self):
        """User and internal buses should be separate."""
        internal_q, user_q = setup_buses()

        send_to_user("user content")
        send_internal("status", "internal event")

        assert user_q.qsize() == 1
        assert internal_q.qsize() == 1

        user_content = user_q.get_nowait()
        internal_event = internal_q.get_nowait()

        assert user_content == "user content"
        assert internal_event["type"] == "status"
        teardown_buses()


class TestPydanticAiIntegration:
    """Test that buses work across pydantic-ai tool execution."""

    @pytest.mark.asyncio
    async def test_user_bus_works_across_pydantic_ai_tools(self):
        """User bus should receive content sent from pydantic-ai tools."""
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel

        # Set up buses BEFORE agent runs
        internal_q, user_q = setup_buses()

        test_agent = Agent(TestModel())

        @test_agent.tool_plain
        async def tool_that_sends_to_user() -> str:
            """Tool that sends content to user."""
            send_to_user("## Rankings from tool")
            return "Rankings sent."

        await test_agent.run("Call tool")

        # Collect user content
        content = collect_user_content(user_q)

        assert len(content) == 1
        assert content[0] == "## Rankings from tool"
        teardown_buses()

    @pytest.mark.asyncio
    async def test_internal_bus_works_across_pydantic_ai_tools(self):
        """Internal bus should receive events from pydantic-ai tools."""
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel

        internal_q, user_q = setup_buses()

        test_agent = Agent(TestModel())

        @test_agent.tool_plain
        async def tool_that_sends_status() -> str:
            """Tool that sends status."""
            send_internal("progress", "50% complete")
            return "Done."

        await test_agent.run("Call tool")

        # Check internal queue
        events = []
        while not internal_q.empty():
            events.append(internal_q.get_nowait())

        progress_events = [e for e in events if e.get("type") == "progress"]
        assert len(progress_events) == 1
        assert "50%" in progress_events[0]["message"]
        teardown_buses()


class TestSSEIntegration:
    """Integration tests for the SSE endpoint."""

    @pytest.mark.asyncio
    async def test_sse_combines_user_content_with_master_response(self):
        """SSE endpoint should combine user bus content with master's response."""
        import json
        from dxtr.server import chat_stream
        from dxtr import data_models

        class MockResult:
            def __init__(self):
                self.output = "Let me know if you need more info!"

            def all_messages(self):
                return []

        async def mock_run_agent(*args, **kwargs):
            send_to_user("## Rankings\n\n**[10/10]** Amazing Paper")
            return MockResult()

        request = data_models.MasterRequest(
            user_id="test-user",
            session_id="test-session-sse",
            query="rank papers",
        )

        with patch("dxtr.server.run_agent", mock_run_agent):
            with patch(
                "dxtr.server.load_session_state",
                AsyncMock(return_value=data_models.SessionState()),
            ):
                response = await chat_stream(request, _token=None)

                events = []
                async for chunk in response.body_iterator:
                    events.append(chunk)

        # Find the done event
        done_event = None
        for event in events:
            if "event: done" in event:
                done_event = event
                break

        assert done_event is not None

        # Parse the answer
        data_line = [l for l in done_event.split("\n") if l.startswith("data:")][0]
        data_json = json.loads(data_line[5:].strip())
        answer = data_json["answer"]

        # Answer should have rankings first, then master's followup
        assert "Rankings" in answer
        assert "Amazing Paper" in answer
        assert "Let me know" in answer
        # Rankings should come before followup
        assert answer.index("Rankings") < answer.index("Let me know")


class TestRankDailyPapers:
    """Tests for the rank_daily_papers tool."""

    @pytest.mark.asyncio
    async def test_rank_daily_papers_sends_to_user_bus(self):
        """rank_daily_papers should send rankings via user bus."""
        from dxtr.agents.master import rank_daily_papers, RankPapersRequest
        from dxtr import set_session_state
        from dxtr.data_models import SessionState

        internal_q, user_q = setup_buses()

        set_session_state(
            SessionState(
                has_synthesized_profile=True,
                profile_content="User interested in ML and NLP.",
            )
        )

        mock_ctx = MagicMock()
        mock_ctx.deps.user_id = "test-user"

        mock_papers = {
            "2601.12345": {
                "id": "2601.12345",
                "title": "Test Paper",
                "abstract": "Abstract",
            },
        }

        mock_ranking_result = [
            {"id": "2601.12345", "title": "Test Paper", "score": 9, "reason": "Relevant"}
        ]

        with patch("dxtr.agents.master.PostgresHelper") as mock_db:
            mock_db.return_value.get_papers_for_ranking.return_value = mock_papers
            with patch(
                "dxtr.agents.subagents.papers_ranking.rank_papers_parallel",
                AsyncMock(return_value=mock_ranking_result),
            ):
                request = RankPapersRequest(date="2026-01-29")
                result = await rank_daily_papers(mock_ctx, request)

        # Check tool return (for master)
        assert "Ranked" in result
        assert "followup" in result.lower()

        # Check user bus received rankings
        content = collect_user_content(user_q)
        assert len(content) == 1
        assert "Test Paper" in content[0] or "2601.12345" in content[0]

        teardown_buses()
