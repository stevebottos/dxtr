"""Tests for the internal message bus system.

Tests the internal bus used for SSE status events (tool progress, status updates).
"""

import asyncio
import pytest

from dxtr.bus import setup_bus, teardown_bus, send_internal


class TestBusBasics:
    """Test basic internal bus functionality."""

    def test_send_internal_puts_event_on_queue(self):
        """send_internal should put event on the internal queue."""
        internal_q = setup_bus()

        send_internal("tool", "Processing started")

        assert not internal_q.empty()
        event = internal_q.get_nowait()
        assert event["type"] == "tool"
        assert event["message"] == "Processing started"
        teardown_bus()

    def test_multiple_events(self):
        """Multiple events should be queued in order."""
        internal_q = setup_bus()

        send_internal("status", "First")
        send_internal("progress", "Second")
        send_internal("tool", "Third")

        events = []
        while not internal_q.empty():
            events.append(internal_q.get_nowait())

        assert len(events) == 3
        assert events[0]["message"] == "First"
        assert events[1]["message"] == "Second"
        assert events[2]["message"] == "Third"
        teardown_bus()


class TestPydanticAiIntegration:
    """Test that internal bus works across pydantic-ai tool execution."""

    @pytest.mark.asyncio
    async def test_internal_bus_works_across_pydantic_ai_tools(self):
        """Internal bus should receive events from pydantic-ai tools."""
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel

        internal_q = setup_bus()

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
        teardown_bus()
