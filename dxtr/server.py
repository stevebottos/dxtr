import asyncio
import json
import os
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
    get_model_settings,
    run_agent,
    create_event_queue,
    clear_event_queue,
)
from dxtr.agents.master import agent as main_agent
from dxtr import data_models

# =============================================================================
# AUTHENTICATION (OIDC)
# =============================================================================

security = HTTPBearer(auto_error=False)

# Configuration from environment
OIDC_ISSUER = os.environ.get("OIDC_ISSUER")
OIDC_AUDIENCE = os.environ.get("OIDC_AUDIENCE")
JWKS_URL = os.environ.get("JWKS_URL") or (f"{OIDC_ISSUER}/.well-known/jwks.json" if OIDC_ISSUER else None)

# PyJWKClient handles caching of the JWKS
jwks_client = jwt.PyJWKClient(JWKS_URL) if JWKS_URL else None


async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify the OIDC token if configuration is present."""
    # Skip verification if not configured (useful for local development)
    if not OIDC_ISSUER or not OIDC_AUDIENCE or not jwks_client:
        return None

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
        )

    try:
        signing_key = jwks_client.get_signing_key_from_jwt(credentials.credentials)
        payload = jwt.decode(
            credentials.credentials,
            signing_key.key,
            algorithms=["RS256"],
            audience=OIDC_AUDIENCE,
            issuer=OIDC_ISSUER,
        )
        return payload
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal auth error",
        )


# =============================================================================
# MEMORY (simple session-based message history)
# =============================================================================

# TODO: Try mem0 for long-term semantic memory across sessions
# https://docs.mem0.ai - extracts facts, does semantic search, persists user context

# For now: simple in-memory session store (swap for Redis in production)
_sessions: dict[str, list[ModelMessage]] = {}


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

# Don't think we'll ever need this again
# @api.post("/chat", response_model=data_models.data_models.MasterResponse)
# async def chat(request: data_models.MasterRequest):
#     answer = await handle_query(request.query, request.user_id, request.session_id)
#     return data_models.MasterResponse(answer=answer)


async def handle_query(request: data_models.MasterRequest) -> str:
    """Process a query through the main agent with conversation history."""
    session_key = get_session_key(request.user_id, request.session_id)

    # Set session context for LiteLLM tracing
    set_session_id(request.session_id)

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

        # Synthetic acknowledgment so user sees immediate feedback
        yield f"event: status\ndata: {json.dumps({'type': 'status', 'message': 'Working on it...'})}\n\n"

        # Run agent in background task
        agent_task = asyncio.create_task(handle_query(request))

        try:
            while not agent_task.done():
                try:
                    # Wait for events with timeout so we can check if agent is done
                    event = await asyncio.wait_for(queue.get(), timeout=0.1)
                    yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    continue

            # Drain any remaining events
            while not queue.empty():
                event = await queue.get()
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

            # Get final result and send as done event
            answer = await agent_task
            yield f"event: done\ndata: {json.dumps({'type': 'done', 'answer': answer})}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
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
