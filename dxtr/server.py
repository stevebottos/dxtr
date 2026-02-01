import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import List, Tuple

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from redis.asyncio import Redis, from_url

from dxtr import constants, data_models, run_agent
from dxtr.agents.master import agent as main_agent
from dxtr.bus import setup_bus, teardown_bus
from dxtr.db import PostgresHelper, close_pool

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


# TODO: What is this?
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


_redis: Redis | None = None


def get_redis() -> Redis:
    """Lazy initialization of Redis client."""
    global _redis
    if _redis is None:
        _redis = from_url(
            os.environ["REDIS_URL"],
            decode_responses=True,
        )
    return _redis


SessionKey = Tuple[str, str]  # (user_id, session_id)


# We use the adapter to handle the "Union" of different message types
def _serialize(msg: ModelMessage) -> str:
    # We wrap the single message in a list because the adapter
    # is optimized for sequences of messages
    return ModelMessagesTypeAdapter.dump_json([msg]).decode("utf-8")


def _deserialize(raw: str) -> ModelMessage:
    # This correctly identifies if the JSON is a ModelRequest or ModelResponse
    # and returns the single message from the decoded list
    return ModelMessagesTypeAdapter.validate_json(raw)[0]


class RedisConversationStore:
    def __init__(
        self,
        redis: Redis,
        *,
        max_history: int = 100,
        ttl_seconds: int | None = 60 * 60 * 24,  # 24h TTL
    ):
        self.redis = redis
        self.max_history = max_history
        self.ttl_seconds = ttl_seconds

    def _key(self, session_key: SessionKey) -> str:
        user_id, session_id = session_key
        return f"chat:{user_id}:{session_id}"

    async def get_history(self, session_key: SessionKey) -> List[ModelMessage]:
        raw = await self.redis.lrange(self._key(session_key), 0, -1)
        return [_deserialize(m) for m in raw]

    async def append(
        self,
        session_key: SessionKey,
        messages: List[ModelMessage],
    ) -> None:
        if not messages:
            return

        key = self._key(session_key)
        pipe = self.redis.pipeline(transaction=True)

        for msg in messages:
            print(msg)
            pipe.rpush(key, _serialize(msg))

        # Keep only the most recent N messages
        pipe.ltrim(key, -self.max_history, -1)

        # Refresh TTL on activity
        if self.ttl_seconds:
            pipe.expire(key, self.ttl_seconds)

        await pipe.execute()

    async def clear_session(self, session_key: SessionKey) -> None:
        """Deletes the history for a specific session."""
        await self.redis.delete(self._key(session_key))

    async def clear_all_users(self) -> None:
        """Deletes all keys starting with 'chat:'"""
        keys = await self.redis.keys("chat:*")
        if keys:
            await self.redis.delete(*keys)


_store: RedisConversationStore | None = None


def get_store() -> RedisConversationStore:
    """Lazy initialization of conversation store."""
    global _store
    if _store is None:
        _store = RedisConversationStore(get_redis())
    return _store


async def dev_nuke_redis():
    """
    DANGER: Wipes the entire current Redis database.
    Only use this during local development.
    """
    await get_redis().flushdb()


def get_user_add_context(user_id: str) -> data_models.AddContext:
    """Build AddContext for a user - can be mocked in tests."""
    db = PostgresHelper()
    facts = db.get_user_facts(user_id)

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
) -> data_models.MasterResponse:
    """
    Process a query through the main agent with Redis-backed
    conversation history.

    No session locks are used:
    - Redis guarantees atomic writes
    - Handlers are stateless
    - Safe across async tasks and multiple workers
    """

    session_key = (request.user_id, request.session_id)
    store = get_store()

    history = await store.get_history(session_key)
    deps = data_models.AgentDeps(request=request, context=add_context)
    result = await run_agent(
        main_agent,
        request.query,
        deps=deps,
        message_history=history,
    )

    # Persist only newly generated messages (no duplication)
    await store.append(
        session_key,
        result.new_messages(),
    )
    return result


# TODO: _token is unused, why?
# TODO: We should rename this to just /chat, and update the front end accordingly (LOW prio)
@api.post("/chat/stream")
async def chat_stream(
    request: data_models.MasterRequest, _token: dict = Depends(verify_token)
):
    """SSE streaming endpoint - sends events as the agent works."""
    # Get context before starting the agent (can be mocked in tests)
    add_context = get_user_add_context(request.user_id)

    async def event_generator():
        # Set up internal bus for status events
        internal_queue = setup_bus()
        last_send = time.time()

        # Immediate acknowledgment
        yield f"event: status\ndata: {json.dumps({'type': 'status', 'message': 'Working on it...'})}\n\n"
        last_send = time.time()

        agent_task = asyncio.create_task(handle_query(request, add_context))

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

            response = data_models.MasterResponse(
                message=response.output,
                artifacts=[],
            )

            # Build response payload
            done_payload = {
                "type": "done",
                "message": response.message,
                "artifacts": [a.model_dump() for a in response.artifacts],
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
