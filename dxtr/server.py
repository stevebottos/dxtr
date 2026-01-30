import asyncio
from collections import defaultdict
import json
import os
import time
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic_ai.messages import ModelMessage

from dxtr import set_session_id, set_session_state, get_model_settings, run_agent
from dxtr.bus import setup_buses, teardown_buses, collect_user_content
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

# For now: simple in-memory session store (swap for Redis in production)
_sessions: dict[str, list[ModelMessage]] = {}
_session_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


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
    close_pool()


api = FastAPI(title="Multi-Agent Server", lifespan=lifespan)

# CORS configuration
origins = [
    "https://dxtrchat.app",
    "https://www.dxtrchat.app",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
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
    """Process a query through the main agent with conversation history."""
    session_key = get_session_key(request.user_id, request.session_id)

    set_session_id(request.session_id)

    async with _session_locks[session_key]:
        state = await load_session_state(request.user_id)
        set_session_state(state)

        history = _sessions.get(session_key, [])

        result = await run_agent(
            main_agent,
            request.query,
            deps=request,
            message_history=history,
            model_settings=get_model_settings(),
        )

        _sessions[session_key] = result.all_messages()

    return result.output


@api.post("/chat/stream")
async def chat_stream(
    request: data_models.MasterRequest, _token: dict = Depends(verify_token)
):
    """SSE streaming endpoint - sends events as the agent works."""

    async def event_generator():
        # Set up both buses for this request
        internal_queue, user_queue = setup_buses()
        last_send = time.time()
        KEEPALIVE_INTERVAL = 10

        # Immediate acknowledgment
        yield f"event: status\ndata: {json.dumps({'type': 'status', 'message': 'Working on it...'})}\n\n"
        last_send = time.time()

        agent_task = asyncio.create_task(handle_query(request))

        try:
            # Stream internal bus events while agent runs
            while not agent_task.done():
                try:
                    event = await asyncio.wait_for(internal_queue.get(), timeout=0.5)
                    yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                    last_send = time.time()
                except asyncio.TimeoutError:
                    if time.time() - last_send >= KEEPALIVE_INTERVAL:
                        yield f"event: status\ndata: {json.dumps({'type': 'status', 'message': 'Still working...'})}\n\n"
                        last_send = time.time()
                    continue

            # Drain remaining internal events
            while not internal_queue.empty():
                event = internal_queue.get_nowait()
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

            # Get master's response
            master_response = await agent_task

            # Collect user bus content (e.g., rankings)
            user_content = collect_user_content(user_queue)

            # Build final answer: user content first, then master's followup
            if user_content:
                answer = "\n\n".join(user_content + [master_response])
            else:
                answer = master_response

            yield f"event: done\ndata: {json.dumps({'type': 'done', 'answer': answer})}\n\n"

        except asyncio.CancelledError:
            agent_task.cancel()
            raise
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            if not agent_task.done():
                agent_task.cancel()
                try:
                    await agent_task
                except asyncio.CancelledError:
                    pass
            teardown_buses()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@api.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("server:api", host="0.0.0.0", port=8000, reload=True)
