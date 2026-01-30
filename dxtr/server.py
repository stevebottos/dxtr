import asyncio
from collections import defaultdict
import json
import os
import time
import uvicorn
from contextlib import asynccontextmanager

import jwt
from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage

from dxtr import (
    set_session_id,
    set_session_state,
    get_model_settings,
    run_agent,
    create_event_queue,
    clear_event_queue,
    was_direct_response_sent,
)
from dxtr.agents.master import agent as main_agent
from dxtr.agents.util import load_session_state
from dxtr import data_models
from dxtr.db import close_pool

# =============================================================================
# AUTHENTICATION (Static API Key)
# =============================================================================

security = HTTPBearer(auto_error=False)


async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify the static DXTR_API_KEY if set in environment."""
    expected_key = os.environ.get("DXTR_API_KEY")
    
    # Skip verification if no key is configured (local dev mode)
    if not expected_key:
        return None

    if not credentials or credentials.credentials != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    
    return {"user": "admin"}


# =============================================================================
# MEMORY (simple session-based message history)
# =============================================================================

# TODO: Try mem0 for long-term semantic memory across sessions
# https://docs.mem0.ai - extracts facts, does semantic search, persists user context

# For now: simple in-memory session store (swap for Redis in production)
_sessions: dict[str, list[ModelMessage]] = {}
_session_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


# =============================================================================
# REQUEST HANDLING
# =============================================================================


def get_session_key(user_id: str, session_id: str) -> str:
    return f"{user_id}:{session_id}"


# =============================================================================
# FASTAPI SERVER
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Multi-agent system ready")
    yield
    print("Shutting down")
    close_pool()  # Clean up database connections


api = FastAPI(title="Multi-Agent Server", lifespan=lifespan)

# CORS configuration
origins = [
    "https://dxtrchat.app",
    "https://www.dxtrchat.app",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",  # Common Vite port
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

api.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def handle_query(request: data_models.MasterRequest) -> str:
    """Process a query through the main agent with conversation history.

    Session state loading and history updates are inside the lock to prevent
    race conditions when concurrent requests arrive for the same session.
    """
    session_key = get_session_key(request.user_id, request.session_id)

    # Set session context for LiteLLM tracing
    set_session_id(request.session_id)

    # Lock per session to prevent concurrent requests from corrupting history
    # IMPORTANT: State loading must be inside the lock to avoid race conditions
    # where two requests load stale state and overwrite each other's changes
    async with _session_locks[session_key]:
        # Load user state from GCS (inside lock to prevent race)
        state = await load_session_state(request.user_id)
        set_session_state(state)

        # Get existing conversation history for this session
        history = _sessions.get(session_key, [])

        # Run agent with message history (streams to console in debug mode)
        result = await run_agent(
            main_agent,
            request.query,
            deps=request,
            message_history=history,
            model_settings=get_model_settings(),
        )

        # Store updated history
        _sessions[session_key] = result.all_messages()

    return result.output


@api.post("/chat/stream")
async def chat_stream(
    request: data_models.MasterRequest,
    _token: dict = Depends(verify_token)
):
    """SSE streaming endpoint - sends events as the agent works."""

    async def event_generator():
        # Create event queue for this request
        queue = create_event_queue()
        last_send = time.time()
        KEEPALIVE_INTERVAL = 10  # seconds

        # Synthetic acknowledgment so user sees immediate feedback
        yield f"event: status\ndata: {json.dumps({'type': 'status', 'message': 'Working on it...'})}\n\n"
        last_send = time.time()

        # Run agent in background task
        agent_task = asyncio.create_task(handle_query(request))

        try:
            while not agent_task.done():
                try:
                    # Wait for events with timeout so we can check if agent is done
                    event = await asyncio.wait_for(queue.get(), timeout=0.5)
                    yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                    last_send = time.time()
                except asyncio.TimeoutError:
                    # Send keepalive if nothing sent recently
                    if time.time() - last_send >= KEEPALIVE_INTERVAL:
                        yield f"event: status\ndata: {json.dumps({'type': 'status', 'message': 'Still working...'})}\n\n"
                        last_send = time.time()
                    continue

            # Drain any remaining events
            while not queue.empty():
                event = await queue.get()
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

            # Get final result
            answer = await agent_task

            # === Handle the three response scenarios (see subagent_response_problem.md) ===
            # Scenario 1 & 2: Master's response is the answer
            # Scenario 3: Direct response was already sent via "content" event, suppress master's echo
            if was_direct_response_sent():
                # Content already streamed to user, send empty done to signal completion
                yield f"event: done\ndata: {json.dumps({'type': 'done', 'answer': ''})}\n\n"
            else:
                yield f"event: done\ndata: {json.dumps({'type': 'done', 'answer': answer})}\n\n"

        except asyncio.CancelledError:
            # Client disconnected - cancel the agent task to free resources
            agent_task.cancel()
            raise
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            # Cancel agent task if still running (client disconnect or error)
            if not agent_task.done():
                agent_task.cancel()
                try:
                    await agent_task
                except asyncio.CancelledError:
                    pass
            clear_event_queue()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@api.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("server:api", host="0.0.0.0", port=8000, reload=True)
