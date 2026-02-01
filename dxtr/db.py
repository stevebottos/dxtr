"""Database helpers for Postgres and Redis."""

import os
from contextlib import contextmanager
from typing import Any, List, Tuple

from pydantic import BaseModel
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from redis.asyncio import Redis, from_url

# =============================================================================
# Postgres
# =============================================================================

_pool: ThreadedConnectionPool | None = None


def _get_pool() -> ThreadedConnectionPool:
    """Get or create the connection pool (lazy initialization)."""
    global _pool
    if _pool is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL not set")
        _pool = ThreadedConnectionPool(minconn=1, maxconn=10, dsn=database_url)
    return _pool


def close_pool():
    """Close all connections in the pool. Call on shutdown."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None


@contextmanager
def _get_connection():
    """Get a connection from the pool, automatically returned when done."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


class PostgresHelper(BaseModel):
    """Simple Postgres helper with raw SQL execution.

    Usage:
        db = PostgresHelper(is_dev=True)
        papers = db.query("SELECT * FROM papers WHERE date = %s", (date,))
        db.execute("DELETE FROM papers WHERE id = %s", (id,))
        new_id = db.execute_returning("INSERT INTO ... RETURNING id", (name,))
    """

    is_dev: bool = False

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute a SELECT query and return results as list of dicts."""
        with _get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(sql, params)
            rows = cur.fetchall()
            cur.close()
            return [dict(row) for row in rows]

    def execute(self, sql: str, params: tuple = ()) -> int:
        """Execute INSERT/UPDATE/DELETE and return affected row count."""
        with _get_connection() as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            count = cur.rowcount
            conn.commit()
            cur.close()
            return count

    def execute_returning(self, sql: str, params: tuple = ()) -> Any:
        """Execute INSERT...RETURNING and return the value."""
        with _get_connection() as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            result = cur.fetchone()[0]
            conn.commit()
            cur.close()
            return result

    @property
    def facts_table(self) -> str:
        """User facts table name (dev or prod)."""
        return "dev_user_facts" if self.is_dev else "user_facts"

    @property
    def rankings_table(self) -> str:
        """Paper rankings table name (dev or prod)."""
        return "dev_user_paper_rankings" if self.is_dev else "user_paper_rankings"


# =============================================================================
# Redis
# =============================================================================

_redis: Redis | None = None


def _get_redis() -> Redis:
    """Get or create the Redis client (lazy initialization)."""
    global _redis
    if _redis is None:
        _redis = from_url(os.environ["REDIS_URL"], decode_responses=True)
    return _redis


SessionKey = Tuple[str, str]  # (user_id, session_id)


def _serialize_message(msg: ModelMessage) -> str:
    return ModelMessagesTypeAdapter.dump_json([msg]).decode("utf-8")


def _deserialize_message(raw: str) -> ModelMessage:
    return ModelMessagesTypeAdapter.validate_json(raw)[0]


class RedisConversationStore:
    """Redis-backed conversation history storage."""

    def __init__(
        self,
        max_history: int = 100,
        ttl_seconds: int | None = 60 * 60 * 24,
    ):
        self.redis = _get_redis()
        self.max_history = max_history
        self.ttl_seconds = ttl_seconds

    def _key(self, session_key: SessionKey) -> str:
        user_id, session_id = session_key
        return f"chat:{user_id}:{session_id}"

    async def get_history(self, session_key: SessionKey) -> List[ModelMessage]:
        raw = await self.redis.lrange(self._key(session_key), 0, -1)
        return [_deserialize_message(m) for m in raw]

    async def append(self, session_key: SessionKey, messages: List[ModelMessage]) -> None:
        if not messages:
            return

        key = self._key(session_key)
        pipe = self.redis.pipeline(transaction=True)

        for msg in messages:
            pipe.rpush(key, _serialize_message(msg))

        pipe.ltrim(key, -self.max_history, -1)

        if self.ttl_seconds:
            pipe.expire(key, self.ttl_seconds)

        await pipe.execute()

    async def clear_session(self, session_key: SessionKey) -> None:
        await self.redis.delete(self._key(session_key))

    async def clear_all(self) -> None:
        keys = await self.redis.keys("chat:*")
        if keys:
            await self.redis.delete(*keys)


# Singleton store instance
_store: RedisConversationStore | None = None


def get_conversation_store() -> RedisConversationStore:
    """Get the singleton conversation store."""
    global _store
    if _store is None:
        _store = RedisConversationStore()
    return _store


async def flush_redis():
    """Flush the entire Redis database. Use for testing only."""
    await _get_redis().flushdb()
