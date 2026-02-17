"""Request/response models for agent interfaces."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class MasterRequest(BaseModel):
    """Incoming chat request."""

    user_id: str
    session_id: str
    query: str


class AddContext(BaseModel):
    """Additional context injected into agent prompts."""

    user_profile_facts: str
    today_date: str
    ranked_dates: list[str] = []
    papers_by_date: dict[str, list[dict]] | None = None  # For testing


class AgentDeps(BaseModel):
    """Dependencies passed to the master agent and its tools."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    request: MasterRequest
    context: AddContext
    db: Any


class PapersRankDeps(BaseModel):
    """Dependencies for the papers ranking agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_id: str
    today_date: str
    user_profile: str
    db: Any
    ranked_dates: list[str] = []
    papers_by_date: dict[str, list[dict]] | None = None  # For testing
