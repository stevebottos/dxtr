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

from dxtr import set_session_id, set_session_state, get_model_settings, run_agent
from dxtr import constants, data_models
from dxtr.bus import setup_bus, teardown_bus
from dxtr.agents.master import agent as main_agent
from dxtr.db import close_pool
from dxtr.storage import get_store, get_session_key

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
# SESSION LOCKS (storage handles data, we just need concurrency control)
# =============================================================================

_session_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


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


async def handle_query(request: data_models.MasterRequest) -> data_models.MasterResponse:
    """Process a query through the main agent with conversation history."""
    from dxtr import get_session_state

    session_key = get_session_key(request.user_id, request.session_id)
    store = get_store()

    set_session_id(request.session_id)

    async with _session_locks[session_key]:
        # Load session state (includes artifact registry)
        state = await store.get_state(session_key)
        set_session_state(state)

        # Get message history
        history = await store.get_history(session_key)

        result = await run_agent(
            main_agent,
            request.query,
            deps=request,
            message_history=history,
            model_settings=get_model_settings(),
        )

        # Save updated message history
        await store.save_history(session_key, result.all_messages())

    # Agent returns plain text
    message = result.output

    # Check session state for artifacts queued for display
    state = get_session_state()
    artifacts = []
    for artifact_id in state.pending_display_artifacts:
        artifact = await store.get_artifact(session_key, artifact_id)
        if artifact:
            artifacts.append(data_models.ArtifactDisplay(
                id=artifact.id,
                content=artifact.content,
                artifact_type=artifact.meta.artifact_type,
            ))

    # Clear pending display for next turn (update stored state)
    if state.pending_display_artifacts:
        state.pending_display_artifacts = []
        await store.save_state(session_key, state)

    return data_models.MasterResponse(
        message=message,
        artifacts=artifacts,
    )


@api.post("/chat/stream")
async def chat_stream(
    request: data_models.MasterRequest, _token: dict = Depends(verify_token)
):
    """SSE streaming endpoint - sends events as the agent works."""

    async def event_generator():
        # Set up internal bus for status events
        internal_queue = setup_bus()
        last_send = time.time()

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
                    if time.time() - last_send >= constants.KEEPALIVE_INTERVAL_SECONDS:
                        yield f"event: status\ndata: {json.dumps({'type': 'status', 'message': 'Still working...'})}\n\n"
                        last_send = time.time()
                    continue

            # Drain remaining internal events
            while not internal_queue.empty():
                event = internal_queue.get_nowait()
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

            # Get agent's response
            response = await agent_task

            # Build response payload
            done_payload = {
                'type': 'done',
                'message': response.message,
                'artifacts': [a.model_dump() for a in response.artifacts],
            }
            yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"

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
            teardown_bus()

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
