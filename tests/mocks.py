"""In-memory replacements for Postgres and Redis so tests only need the LLM."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelMessage

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class InMemoryDB:
    """Drop-in replacement for PostgresHelper backed by plain lists.

    Supports only the SQL patterns actually used in the test journey.
    Raises ValueError on anything unrecognised so failures are loud.
    """

    def __init__(self) -> None:
        self._facts: list[dict] = []
        self._rankings: list[dict] = []
        self._papers: dict[str, list[dict]] = {}
        self._next_id: int = 1
        self._load_papers()

    def _load_papers(self) -> None:
        path = FIXTURES_DIR / "papers.json"
        if path.exists():
            self._papers = json.loads(path.read_text())

    # --- table name properties (mirror PostgresHelper) ---

    @property
    def facts_table(self) -> str:
        return "dev_user_facts"

    @property
    def rankings_table(self) -> str:
        return "dev_user_paper_rankings"

    # --- helpers ---

    def _table_for_sql(self, sql: str) -> str:
        if self.facts_table in sql:
            return "facts"
        if self.rankings_table in sql:
            return "rankings"
        if "papers" in sql.lower():
            return "papers"
        raise ValueError(f"InMemoryDB: unrecognised table in SQL: {sql}")

    # --- public API (same signatures as PostgresHelper) ---

    def _build_papers_index(self) -> dict[str, dict]:
        """Build a lookup from paper id -> paper dict across all dates."""
        return {p["id"]: p for papers in self._papers.values() for p in papers}

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        table = self._table_for_sql(sql)
        if table == "facts":
            user_id = params[0]
            return [f for f in self._facts if f["user_id"] == user_id]
        if table == "papers":
            if "WHERE id" in sql:
                # Single paper lookup by id: params = (paper_id,)
                paper_id = params[0]
                papers_idx = self._build_papers_index()
                paper = papers_idx.get(paper_id)
                return [paper] if paper else []
            # Papers by date: params = (date,)
            date_key = str(params[0])
            return self._papers.get(date_key, [])
        if table == "rankings":
            user_id = params[0]
            # DISTINCT paper_date query
            if "DISTINCT paper_date" in sql:
                dates = sorted(
                    {str(r["paper_date"]) for r in self._rankings if r["user_id"] == user_id},
                    reverse=True,
                )
                return [{"paper_date": d} for d in dates]
            # Single paper ranking lookup: params = (user_id, paper_id, paper_date)
            if "r.paper_id = %s" in sql:
                paper_id = params[1]
                paper_date = str(params[2])
                for r in self._rankings:
                    if (
                        r["user_id"] == user_id
                        and r["paper_id"] == paper_id
                        and str(r["paper_date"]) == paper_date
                        and r["ranking_criteria_type"] == "profile"
                    ):
                        return [{"ranking": r["ranking"], "reason": r["reason"]}]
                return []
            # JOIN rankings with papers: params = (user_id, paper_date)
            paper_date = str(params[1])
            papers_idx = self._build_papers_index()
            results = []
            for r in self._rankings:
                if (
                    r["user_id"] == user_id
                    and str(r["paper_date"]) == paper_date
                    and r["ranking_criteria_type"] == "profile"
                    and r["paper_id"] in papers_idx
                ):
                    p = papers_idx[r["paper_id"]]
                    results.append({
                        "paper_id": r["paper_id"],
                        "ranking": r["ranking"],
                        "reason": r["reason"],
                        "title": p["title"],
                        "summary": p.get("summary", ""),
                        "authors": p.get("authors", []),
                        "upvotes": p.get("upvotes", 0),
                    })
            results.sort(key=lambda x: x["ranking"], reverse=True)
            return results
        raise ValueError(f"InMemoryDB.query: unsupported SQL: {sql}")

    def execute(self, sql: str, params: tuple = ()) -> int:
        table = self._table_for_sql(sql)
        upper = sql.upper().strip()

        if upper.startswith("DELETE"):
            return self._handle_delete(table, params)
        if upper.startswith("INSERT"):
            self._handle_insert(table, sql, params)
            return 1
        raise ValueError(f"InMemoryDB.execute: unsupported SQL: {sql}")

    def execute_returning(self, sql: str, params: tuple = ()) -> Any:
        table = self._table_for_sql(sql)
        upper = sql.upper().strip()

        if upper.startswith("INSERT"):
            return self._handle_insert(table, sql, params)
        raise ValueError(f"InMemoryDB.execute_returning: unsupported SQL: {sql}")

    # --- internal handlers ---

    def _handle_delete(self, table: str, params: tuple) -> int:
        user_id = params[0]
        if table == "facts":
            before = len(self._facts)
            self._facts = [f for f in self._facts if f["user_id"] != user_id]
            return before - len(self._facts)
        if table == "rankings":
            before = len(self._rankings)
            self._rankings = [r for r in self._rankings if r["user_id"] != user_id]
            return before - len(self._rankings)
        raise ValueError(f"InMemoryDB._handle_delete: unknown table {table}")

    def _handle_insert(self, table: str, sql: str, params: tuple) -> int:
        if table == "facts":
            row_id = self._next_id
            self._next_id += 1
            self._facts.append(
                {
                    "id": row_id,
                    "user_id": params[0],
                    "fact": params[1],
                    "created_at": datetime.now(),
                }
            )
            return row_id
        if table == "rankings":
            # params: (user_id, paper_id, paper_date, criteria_type, criteria, ranking, reason)
            self._rankings.append(
                {
                    "user_id": params[0],
                    "paper_id": params[1],
                    "paper_date": params[2],
                    "ranking_criteria_type": params[3],
                    "ranking_criteria": params[4],
                    "ranking": params[5],
                    "reason": params[6],
                }
            )
            return self._next_id  # rankings don't use RETURNING but keep consistent
        raise ValueError(f"InMemoryDB._handle_insert: unknown table {table}")

    def reset(self) -> None:
        """Clear user data (call between tests). Papers fixture is preserved."""
        self._facts.clear()
        self._rankings.clear()
        self._next_id = 1


SessionKey = tuple[str, str]


class InMemoryConversationStore:
    """Drop-in replacement for RedisConversationStore backed by a dict."""

    def __init__(self) -> None:
        self._sessions: dict[SessionKey, list[ModelMessage]] = {}

    async def get_history(self, session_key: SessionKey) -> list[ModelMessage]:
        return list(self._sessions.get(session_key, []))

    async def append(self, session_key: SessionKey, messages: list[ModelMessage]) -> None:
        if not messages:
            return
        self._sessions.setdefault(session_key, []).extend(messages)

    async def clear_session(self, session_key: SessionKey) -> None:
        self._sessions.pop(session_key, None)

    async def clear_all(self) -> None:
        self._sessions.clear()

    def reset(self) -> None:
        """Clear all sessions (call between tests)."""
        self._sessions.clear()
