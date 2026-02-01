"""Request/response models for agent interfaces."""

from pydantic import BaseModel

from dxtr.db import PostgresHelper


class MasterRequest(BaseModel):
    """Incoming chat request."""

    user_id: str
    session_id: str
    query: str


class AddContext(BaseModel):
    """Additional context injected into agent prompts."""

    user_profile_facts: str
    today_date: str
    papers_by_date: dict[str, list[dict]] | None = None  # For testing


class AgentDeps(BaseModel):
    """Dependencies passed to the master agent and its tools."""

    request: MasterRequest
    context: AddContext
    db: PostgresHelper


class PapersRankDeps(BaseModel):
    """Dependencies for the papers ranking agent."""

    date_to_rank: str
    user_profile: str
    db: PostgresHelper
    papers_by_date: dict[str, list[dict]] | None = None  # For testing
