import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage

from dxtr import set_session_id, get_model_settings
from dxtr.agents.master import agent as main_agent


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


async def handle_query(query: str, user_id: str, session_id: str) -> str:
    """Process a query through the main agent with conversation history."""
    session_key = get_session_key(user_id, session_id)

    # Set session context for LiteLLM tracing
    set_session_id(session_id)

    # Get existing conversation history for this session
    history = _sessions.get(session_key, [])

    # Run agent with message history and session metadata
    result = await main_agent.run(
        query,
        message_history=history,
        model_settings=get_model_settings(),
    )

    # Store updated history
    _sessions[session_key] = result.all_messages()

    return result.output


# =============================================================================
# FASTAPI SERVER
# =============================================================================


class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    query: str


class ChatResponse(BaseModel):
    answer: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Multi-agent system ready")
    yield
    print("Shutting down")


api = FastAPI(title="Multi-Agent Server", lifespan=lifespan)


@api.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    answer = await handle_query(request.query, request.user_id, request.session_id)
    return ChatResponse(answer=answer)


@api.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("server:api", host="0.0.0.0", port=8000, reload=True)
