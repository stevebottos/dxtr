import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import date, timedelta

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from dxtr import constants, data_models
from dxtr.agents.master import agent as main_agent
from dxtr.bus import setup_bus, teardown_bus
from dxtr.db import PostgresHelper, close_pool, get_conversation_store, flush_redis

security = HTTPBearer(auto_error=False)

# Shared database helper for production (is_dev=False)
PROD_DB = PostgresHelper(is_dev=False)


async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify the static DXTR_API_KEY if set in environment."""
    expected_key = os.environ.get("DXTR_API_KEY")

    if not expected_key:
        return None

    if not credentials or credentials.credentials != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )

    return {"user": "admin"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Multi-agent system ready")
    yield
    print("Shutting down")
    close_pool()


api = FastAPI(title="Multi-Agent Server", lifespan=lifespan)

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


# For tests
async def dev_nuke_redis():
    """Wipes the entire Redis database. Only use during testing."""
    await flush_redis()


def get_user_add_context(user_id: str, db: PostgresHelper) -> data_models.AddContext:
    """Build AddContext for a user."""
    facts = db.query(
        f"SELECT id, fact, created_at FROM {db.facts_table} WHERE user_id = %s ORDER BY created_at ASC",
        (user_id,),
    )

    if not facts:
        user_profile_facts = "No facts stored about this user yet."
    else:
        lines = [f"Known facts about user ({len(facts)} total):"]
        for f in facts:
            timestamp = f["created_at"].strftime("%Y-%m-%d %H:%M")
            lines.append(f"- [{timestamp}] {f['fact']}")
        user_profile_facts = "\n".join(lines)

    today = date.today()

    # Build a date reference table for the past week (LLMs are bad at date math)
    date_lines = ["Date reference:"]
    for days_ago in range(8):
        d = today - timedelta(days=days_ago)
        label = "today" if days_ago == 0 else "yesterday" if days_ago == 1 else f"{days_ago} days ago"
        date_lines.append(f"  {d.strftime('%A')}: {d.isoformat()} ({label})")
    today_str = "\n".join(date_lines)

    return data_models.AddContext(
        user_profile_facts=user_profile_facts,
        today_date=today_str,
    )


async def handle_query(
    request: data_models.MasterRequest,
    add_context: data_models.AddContext,
    db: PostgresHelper,
):
    """Process a query through the main agent with Redis-backed conversation history."""

    session_key = (request.user_id, request.session_id)
    store = get_conversation_store()

    history = await store.get_history(session_key)
    deps = data_models.AgentDeps(request=request, context=add_context, db=db)
    result = await main_agent.run(request.query, deps=deps, message_history=history)

    await store.append(session_key, result.new_messages())
    return result


@api.post("/chat/stream")
async def chat_stream(
    request: data_models.MasterRequest, _token: dict = Depends(verify_token)
):
    """SSE streaming endpoint - sends events as the agent works."""
    add_context = get_user_add_context(request.user_id, PROD_DB)

    async def event_generator():
        internal_queue = setup_bus()
        last_send = time.time()

        yield f"event: status\ndata: {json.dumps({'type': 'status', 'message': 'Working on it...'})}\n\n"
        last_send = time.time()

        agent_task = asyncio.create_task(handle_query(request, add_context, PROD_DB))

        try:
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

            while not internal_queue.empty():
                event = internal_queue.get_nowait()
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

            result = await agent_task

            done_payload = {"type": "done", "message": result.output}
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


@api.delete("/user/{user_id}/profile")
async def clear_user_profile(user_id: str, _token: dict = Depends(verify_token)):
    """Delete all stored facts for a user."""
    count = PROD_DB.execute(
        f"DELETE FROM {PROD_DB.facts_table} WHERE user_id = %s",
        (user_id,),
    )
    return {"deleted": count}


@api.delete("/user/{user_id}/history/{session_id}")
async def clear_conversation_history(
    user_id: str, session_id: str, _token: dict = Depends(verify_token)
):
    """Clear conversation history for a user's session."""
    store = get_conversation_store()
    await store.clear_session((user_id, session_id))
    return {"cleared": True}


@api.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("server:api", host="0.0.0.0", port=8000, reload=True)
