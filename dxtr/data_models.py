"""
This module defines requests/responses for agent interfaces.
"""

import re

from pydantic import BaseModel, field_validator

# Pattern for safe IDs: UUIDs, alphanumeric, underscores, hyphens
# Prevents path traversal attacks (e.g., ../../../etc)
SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


# These fields are set explicitly via the chat interface
class MasterRequest(BaseModel):
    user_id: str
    session_id: str
    query: str

    @field_validator("user_id", "session_id")
    @classmethod
    def validate_safe_id(cls, v: str) -> str:
        """Ensure IDs are safe for use in file paths (no path traversal)."""
        if not v:
            raise ValueError("ID cannot be empty")
        if len(v) > 128:
            raise ValueError(f"ID too long (max 128 chars), got {len(v)}")
        if not SAFE_ID_PATTERN.match(v):
            raise ValueError(
                f"ID must contain only alphanumeric chars, underscores, or hyphens. Got: {v[:20]}..."
            )
        return v


class AddContext(BaseModel):
    user_profile_facts: str
    today_date: str  # YYYY-MM-DD format
    papers_by_date: dict[str, list[dict]] | None = None  # date -> papers, for testing


class AgentDeps(BaseModel):
    """Combined deps for the master agent - makes testing easier."""

    request: "MasterRequest"
    context: AddContext


class PapersRankDeps(BaseModel):
    """Deps for the papers ranking agent."""

    date_to_rank: str  # YYYY-MM-DD format
    user_profile: str  # User's profile/facts as a string
    papers_by_date: dict[str, list[dict]] | None = None  # date -> papers, for testing


class ArtifactDisplay(BaseModel):
    """Artifact to be displayed by the frontend."""

    id: int
    content: str
    artifact_type: str


class MasterResponse(BaseModel):
    """Response sent to frontend."""

    message: str  # Agent's text response
    artifacts: list[ArtifactDisplay] = []  # Artifacts to display alongside message
