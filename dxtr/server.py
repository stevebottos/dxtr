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
from dxtr.db import PostgresHelper, RedisConversationStore, close_pool, get_conversation_store

security = HTTPBearer(auto_error=False)

# Use dev tables if DEV_USER_ID is set (local development)
IS_DEV = bool(os.environ.get("DEV_USER_ID"))
DB = PostgresHelper(is_dev=IS_DEV)


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
        label = (
            "today"
            if days_ago == 0
            else "yesterday"
            if days_ago == 1
            else f"{days_ago} days ago"
        )
        date_lines.append(f"  {d.strftime('%A')}: {d.isoformat()} ({label})")
    today_str = "\n".join(date_lines)

    ranked_rows = db.query(
        f"SELECT DISTINCT paper_date FROM {db.rankings_table} WHERE user_id = %s ORDER BY paper_date DESC",
        (user_id,),
    )
    ranked_dates = [str(r["paper_date"]) for r in ranked_rows]

    return data_models.AddContext(
        user_profile_facts=user_profile_facts,
        today_date=today_str,
        ranked_dates=ranked_dates,
    )


async def handle_query(
    request: data_models.MasterRequest,
    db: PostgresHelper,
    store: RedisConversationStore | None = None,
):
    """Process a query through the main agent with Redis-backed conversation history."""

    add_context = get_user_add_context(request.user_id, db)

    session_key = (request.user_id, request.session_id)
    store = store or get_conversation_store()

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

    async def event_generator():
        internal_queue = setup_bus()
        last_send = time.time()

        yield f"event: status\ndata: {json.dumps({'type': 'status', 'message': 'Working on it...'})}\n\n"
        last_send = time.time()

        agent_task = asyncio.create_task(handle_query(request, DB))

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
    count = DB.execute(
        f"DELETE FROM {DB.facts_table} WHERE user_id = %s",
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


class DeleteRankingRequest(data_models.BaseModel):
    date: str
    criteria_key: str


@api.delete("/rankings/{user_id}")
async def delete_ranking_group(
    user_id: str, request: DeleteRankingRequest, _token: dict = Depends(verify_token)
):
    """Delete a ranking group by date and criteria key."""
    # criteria_key format: "profile:..." or "request:..."
    criteria_type, criteria_prefix = request.criteria_key.split(":", 1)

    count = DB.execute(
        f"""
        DELETE FROM {DB.rankings_table}
        WHERE user_id = %s
          AND paper_date = %s
          AND ranking_criteria_type = %s
          AND LEFT(ranking_criteria, 50) = %s
        """,
        (user_id, request.date, criteria_type, criteria_prefix),
    )
    return {"deleted": count}


@api.get("/rankings/{user_id}")
async def get_rankings(user_id: str, _token: dict = Depends(verify_token)):
    """Get all rankings for a user, grouped by date and criteria type."""
    rankings = DB.query(
        f"""
        SELECT r.paper_id, r.paper_date, r.ranking_criteria_type, r.ranking_criteria,
               r.ranking, r.reason, r.created_at,
               p.title, p.summary, p.authors, p.upvotes
        FROM {DB.rankings_table} r
        JOIN papers p ON r.paper_id = p.id
        WHERE r.user_id = %s
        ORDER BY r.paper_date DESC, r.ranking DESC
        """,
        (user_id,),
    )

    # Group by date and criteria type
    grouped: dict = {}
    for r in rankings:
        date_key = r["paper_date"].isoformat()
        criteria_key = f"{r['ranking_criteria_type']}:{r['ranking_criteria'][:50]}"

        if date_key not in grouped:
            grouped[date_key] = {}
        if criteria_key not in grouped[date_key]:
            grouped[date_key][criteria_key] = {
                "criteria_type": r["ranking_criteria_type"],
                "criteria": r["ranking_criteria"],
                "papers": [],
            }

        grouped[date_key][criteria_key]["papers"].append(
            {
                "paper_id": r["paper_id"],
                "title": r["title"],
                "summary": r["summary"],
                "authors": r["authors"],
                "upvotes": r["upvotes"],
                "ranking": r["ranking"],
                "reason": r["reason"],
            }
        )

    return {"rankings": grouped}


@api.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("server:api", host="0.0.0.0", port=8000, reload=True)
