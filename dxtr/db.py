"""Database helper for paper queries."""

import os
from datetime import date, timedelta

import psycopg2
from psycopg2.extras import RealDictCursor


class PostgresHelper:
    """Database helper for paper queries."""

    def __init__(self, database_url: str | None = None):
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL not set")

    def _connect(self):
        return psycopg2.connect(self.database_url)

    def get_available_dates(self, days_back: int = 7) -> list[dict]:
        """Get dates that have papers, with counts.

        Returns:
            List of {date: str, count: int} sorted by date descending
        """
        conn = self._connect()
        try:
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
        finally:
            conn.close()

    def get_papers_by_date(self, target_date: date | str) -> list[dict]:
        """Get all papers for a date.

        Args:
            target_date: Date object or string in YYYY-MM-DD format

        Returns:
            List of paper dicts with id, title, summary, upvotes, etc.
        """
        if isinstance(target_date, str):
            target_date = date.fromisoformat(target_date)

        conn = self._connect()
        try:
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
        finally:
            conn.close()

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

        conn = self._connect()
        try:
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
        finally:
            conn.close()

    def get_papers_for_ranking(self, target_date: date | str) -> dict[str, dict]:
        """Get papers in format needed by ranking agent.

        Args:
            target_date: Date object or string in YYYY-MM-DD format

        Returns:
            Dict of {paper_id: {title, summary}} for ranking agent
        """
        papers = self.get_papers_by_date(target_date)
        return {
            p["id"]: {"title": p["title"], "summary": p["summary"]}
            for p in papers
        }

    def get_paper_stats(self, days_back: int = 7) -> dict:
        """Get aggregate statistics about papers.

        Returns:
            Dict with total_papers, days_with_papers, earliest_date, latest_date, avg_upvotes
        """
        conn = self._connect()
        try:
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
        finally:
            conn.close()

    def get_paper_count(self, days_back: int | None = None) -> int:
        """Get total number of papers, optionally filtered by date range.

        Args:
            days_back: If provided, only count papers from the last N days.
                      If None, count all papers.

        Returns:
            Total paper count
        """
        conn = self._connect()
        try:
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
        finally:
            conn.close()

    def get_date_with_most_papers(self, days_back: int = 7) -> dict | None:
        """Get the date that has the most papers.

        Returns:
            Dict with {date, count} or None if no papers
        """
        conn = self._connect()
        try:
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
        finally:
            conn.close()
