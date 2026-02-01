"""Database helper for paper queries with connection pooling."""

import json
import os
from contextlib import contextmanager
from datetime import date, timedelta

from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

# TODO: Why do we have TTL here? Shouldn't this be assigned in the migration/table schema? I don't know, just checking
# Rankings TTL in hours
RANKINGS_TTL_HOURS = 24

# Module-level connection pool (initialized lazily)
_pool: ThreadedConnectionPool | None = None


def _get_pool() -> ThreadedConnectionPool:
    """Get or create the connection pool (lazy initialization)."""
    global _pool
    if _pool is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL not set")
        # minconn=1: always keep at least 1 connection ready
        # maxconn=10: allow up to 10 concurrent connections
        _pool = ThreadedConnectionPool(minconn=1, maxconn=10, dsn=database_url)
    return _pool


def close_pool():
    """Close all connections in the pool. Call on shutdown."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None


@contextmanager
def get_connection():
    """Get a connection from the pool, automatically returned when done.

    Usage:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(...)
    """
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


class PostgresHelper:
    """Database helper for paper queries.

    Uses a shared connection pool for efficient connection management.
    Automatically uses dev tables (e.g., dev_user_facts) when IS_DEV=1.
    """

    def __init__(self):
        self.use_dev_tables = os.getenv("IS_DEV", "1") == "1"

    def get_available_dates(self, days_back: int = 7) -> list[dict]:
        """Get dates that have papers, with counts.

        Returns:
            List of {date: str, count: int} sorted by date descending
        """
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                """
                SELECT date, COUNT(*) as count
                FROM papers
                WHERE date >= %s
                GROUP BY date
                ORDER BY date DESC
                """,
                (date.today() - timedelta(days=days_back),),
            )
            rows = cur.fetchall()
            cur.close()
            return [{"date": str(row["date"]), "count": row["count"]} for row in rows]

    def get_papers_by_date(self, target_date: date | str) -> list[dict]:
        """Get all papers for a date.

        Args:
            target_date: Date object or string in YYYY-MM-DD format

        Returns:
            List of paper dicts with id, title, summary, upvotes, etc.
        """
        if isinstance(target_date, str):
            target_date = date.fromisoformat(target_date)

        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                """
                SELECT id, title, summary, authors, published_at, upvotes, date
                FROM papers
                WHERE date = %s
                ORDER BY upvotes DESC
                """,
                (target_date,),
            )
            rows = cur.fetchall()
            cur.close()
            return [dict(row) for row in rows]

    def get_top_papers(self, target_date: date | str, limit: int = 10) -> list[dict]:
        """Get top N papers by upvotes for a date.

        Args:
            target_date: Date object or string in YYYY-MM-DD format
            limit: Max papers to return

        Returns:
            List of paper dicts sorted by upvotes descending
        """
        if isinstance(target_date, str):
            target_date = date.fromisoformat(target_date)

        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                """
                SELECT id, title, summary, authors, published_at, upvotes, date
                FROM papers
                WHERE date = %s
                ORDER BY upvotes DESC
                LIMIT %s
                """,
                (target_date, limit),
            )
            rows = cur.fetchall()
            cur.close()
            return [dict(row) for row in rows]

    def get_papers_for_ranking(self, target_date: date | str) -> dict[str, dict]:
        """Get papers in format needed by ranking agent.

        Args:
            target_date: Date object or string in YYYY-MM-DD format

        Returns:
            Dict of {paper_id: {title, summary}} for ranking agent
        """
        papers = self.get_papers_by_date(target_date)
        return {p["id"]: {"title": p["title"], "summary": p["summary"]} for p in papers}

    def get_paper_stats(self, days_back: int = 7) -> dict:
        """Get aggregate statistics about papers.

        Returns:
            Dict with total_papers, days_with_papers, earliest_date, latest_date, avg_upvotes
        """
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                """
                SELECT
                    COUNT(*) as total_papers,
                    COUNT(DISTINCT date) as days_with_papers,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date,
                    ROUND(AVG(upvotes)::numeric, 1) as avg_upvotes
                FROM papers
                WHERE date >= %s
                """,
                (date.today() - timedelta(days=days_back),),
            )
            row = cur.fetchone()
            cur.close()
            return dict(row) if row else {}

    def get_paper_count(self, days_back: int | None = None) -> int:
        """Get total number of papers, optionally filtered by date range.

        Args:
            days_back: If provided, only count papers from the last N days.
                      If None, count all papers.

        Returns:
            Total paper count
        """
        with get_connection() as conn:
            cur = conn.cursor()
            if days_back is not None:
                cur.execute(
                    "SELECT COUNT(*) FROM papers WHERE date >= %s",
                    (date.today() - timedelta(days=days_back),),
                )
            else:
                cur.execute("SELECT COUNT(*) FROM papers")
            count = cur.fetchone()[0]
            cur.close()
            return count

    def get_date_with_most_papers(self, days_back: int = 7) -> dict | None:
        """Get the date that has the most papers.

        Returns:
            Dict with {date, count} or None if no papers
        """
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                """
                SELECT date, COUNT(*) as count
                FROM papers
                WHERE date >= %s
                GROUP BY date
                ORDER BY count DESC
                LIMIT 1
                """,
                (date.today() - timedelta(days=days_back),),
            )
            row = cur.fetchone()
            cur.close()
            if row:
                return {"date": str(row["date"]), "count": row["count"]}
            return None

    # === Rankings Storage ===

    def get_rankings(self, user_id: str, paper_date: str) -> list[dict] | None:
        """Get cached rankings for a user and date.

        Args:
            user_id: User identifier
            paper_date: Date string (YYYY-MM-DD) for the papers ranked

        Returns:
            List of ranked papers or None if not cached/expired
        """
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                """
                SELECT rankings FROM user_rankings
                WHERE user_id = %s AND date = %s AND expires_at > NOW()
                """,
                (user_id, paper_date),
            )
            row = cur.fetchone()
            cur.close()
            if row:
                return row["rankings"]
            return None

    def save_rankings(
        self, user_id: str, paper_date: str, rankings: list[dict]
    ) -> None:
        """Save rankings for a user and date with TTL.

        Args:
            user_id: User identifier
            paper_date: Date string (YYYY-MM-DD) for the papers ranked
            rankings: List of ranked paper dicts
        """
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                f"""
                INSERT INTO user_rankings (user_id, date, rankings, expires_at)
                VALUES (%s, %s, %s, NOW() + INTERVAL '{RANKINGS_TTL_HOURS} hours')
                ON CONFLICT (user_id, date) DO UPDATE
                SET rankings = EXCLUDED.rankings,
                    expires_at = NOW() + INTERVAL '{RANKINGS_TTL_HOURS} hours'
                """,
                (user_id, paper_date, json.dumps(rankings)),
            )
            conn.commit()
            cur.close()

    def cleanup_expired_rankings(self) -> int:
        """Delete expired rankings. Returns number of rows deleted."""
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM user_rankings WHERE expires_at < NOW()")
            deleted = cur.rowcount
            conn.commit()
            cur.close()
            return deleted

    # === User Facts Storage ===

    def _facts_table(self) -> str:
        """Get the appropriate user facts table name."""
        return "dev_user_facts" if self.use_dev_tables else "user_facts"

    def store_user_fact(self, user_id: str, fact: str) -> int:
        """Store a fact about a user.

        Args:
            user_id: User identifier
            fact: The fact to store

        Returns:
            The ID of the inserted fact
        """
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                f"""
                INSERT INTO {self._facts_table()} (user_id, fact)
                VALUES (%s, %s)
                RETURNING id
                """,
                (user_id, fact),
            )
            fact_id = cur.fetchone()[0]
            conn.commit()
            cur.close()
            return fact_id

    def get_user_facts(self, user_id: str) -> list[dict]:
        """Get all facts for a user in chronological order.

        Args:
            user_id: User identifier

        Returns:
            List of {id, fact, created_at} dicts ordered by created_at ascending
        """
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                f"""
                SELECT id, fact, created_at
                FROM {self._facts_table()}
                WHERE user_id = %s
                ORDER BY created_at ASC
                """,
                (user_id,),
            )
            rows = cur.fetchall()
            cur.close()
            return [dict(row) for row in rows]

    def query(self, query: str):
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(query)
            rows = cur.fetchall()
            cur.close()
            return rows

    def delete_user_facts(self, user_id: str) -> int:
        """Delete all facts for a user. Useful for testing.

        Args:
            user_id: User identifier

        Returns:
            Number of facts deleted
        """
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                f"DELETE FROM {self._facts_table()} WHERE user_id = %s", (user_id,)
            )
            deleted = cur.rowcount
            conn.commit()
            cur.close()
            return deleted

    # === Paper Rankings Storage ===

    def _rankings_table(self) -> str:
        """Get the appropriate paper rankings table name."""
        return (
            "dev_user_paper_rankings" if self.use_dev_tables else "user_paper_rankings"
        )

    def save_paper_ranking(
        self,
        user_id: str,
        paper_date: str,
        paper_id: str,
        ranking: int,
        reason: str,
        user_query: str | None = None,
    ) -> int:
        """Save a paper ranking for a user.

        Args:
            user_id: User identifier
            paper_date: Date of the paper (YYYY-MM-DD)
            paper_id: Paper identifier (e.g., arXiv ID)
            ranking: Score 1-5
            reason: Reason for the ranking
            user_query: Optional user query that triggered the ranking

        Returns:
            The ID of the inserted/updated ranking
        """
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                f"""
                INSERT INTO {self._rankings_table()}
                    (user_id, paper_date, paper_id, ranking, reason, user_query)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, paper_date, paper_id) DO UPDATE
                SET ranking = EXCLUDED.ranking,
                    reason = EXCLUDED.reason,
                    user_query = EXCLUDED.user_query,
                    created_at = NOW()
                RETURNING id
                """,
                (user_id, paper_date, paper_id, ranking, reason, user_query),
            )
            ranking_id = cur.fetchone()[0]
            conn.commit()
            cur.close()
            return ranking_id

    def get_paper_rankings(self, user_id: str, paper_date: str) -> list[dict]:
        """Get all paper rankings for a user and date.

        Args:
            user_id: User identifier
            paper_date: Date of the papers (YYYY-MM-DD)

        Returns:
            List of {paper_id, ranking, reason, created_at} dicts ordered by ranking descending
        """
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                f"""
                SELECT paper_id, ranking, reason, created_at
                FROM {self._rankings_table()}
                WHERE user_id = %s AND paper_date = %s
                ORDER BY ranking DESC
                """,
                (user_id, paper_date),
            )
            rows = cur.fetchall()
            cur.close()
            return [dict(row) for row in rows]

    def delete_paper_rankings(self, user_id: str, paper_date: str | None = None) -> int:
        """Delete paper rankings for a user. Useful for testing.

        Args:
            user_id: User identifier
            paper_date: Optional date to filter by. If None, deletes all rankings for user.

        Returns:
            Number of rankings deleted
        """
        with get_connection() as conn:
            cur = conn.cursor()
            if paper_date:
                cur.execute(
                    f"DELETE FROM {self._rankings_table()} WHERE user_id = %s AND paper_date = %s",
                    (user_id, paper_date),
                )
            else:
                cur.execute(
                    f"DELETE FROM {self._rankings_table()} WHERE user_id = %s",
                    (user_id,),
                )
            deleted = cur.rowcount
            conn.commit()
            cur.close()
            return deleted
