"""
This module defines requests/responses for agent interfaces.
"""

import re
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

# Pattern for safe IDs: UUIDs, alphanumeric, underscores, hyphens
# Prevents path traversal attacks (e.g., ../../../etc)
SAFE_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')


def validate_date_format(v: str) -> str:
    """Validate date is in YYYY-MM-DD format."""
    try:
        datetime.strptime(v, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Date must be in YYYY-MM-DD format, got: {v}")
    return v


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


class ArtifactDisplay(BaseModel):
    """Artifact to be displayed by the frontend."""
    id: int
    content: str
    artifact_type: str


class MasterResponse(BaseModel):
    """Response sent to frontend."""
    message: str  # Agent's text response
    artifacts: list[ArtifactDisplay] = []  # Artifacts to display alongside message


# These fields are set implicitly by the LLM, so we have descriptors for them
class ProfileSynthesisRequest(MasterRequest):
    with_github_summary: bool = Field(
        description="""Whether or not to pull a previously computed github summary from the user's profile for synthesis.""",
    )


class GithubSummarizerRequest(MasterRequest):
    repo_urls: list[str] = Field(
        description="""A COMPLETE list of all GitHub repository URLs identified in the conversation.
        Collect every relevant URL into this single list before calling the tool.",
        format: https://github.com/<username>/<repo-name> (with or without .git)
        """,
        examples=[
            [
                "https://github.com/stevebottos/chess.git",
                "https://github.com/anthropics/claude-code",
            ],
            ["https://github.com/google/google/research"],
        ],
    )

    @field_validator("repo_urls")
    @classmethod
    def ensure_git_suffix(cls, urls: list[str]) -> list[str]:
        cleaned = []
        for url in urls:
            url = url.strip().rstrip("/")
            if not url.endswith(".git"):
                url += ".git"
            cleaned.append(url)
        return cleaned


class GithubSummarizerResponse(BaseModel):
    github_summary: str = Field(
        description="The JSON summary from GitHub analysis.",
    )


# SessionState is now in storage.py for clean separation
# Re-export for backwards compatibility
from dxtr.storage import SessionState, ArtifactMeta, ArtifactType


# === Paper Tool Request Models ===


class GetPapersRequest(BaseModel):
    days_back: int = Field(
        default=7,
        description="Number of days to look back for available papers.",
    )


class FetchPapersRequest(BaseModel):
    date: str = Field(
        description="Date to fetch papers for (format: YYYY-MM-DD).",
        examples=["2024-01-15"],
    )
    _validate_date = field_validator("date")(validate_date_format)


class DownloadPapersRequest(BaseModel):
    date: str = Field(
        description="Date to download papers for (format: YYYY-MM-DD).",
        examples=["2024-01-15"],
    )
    paper_ids: list[str] | None = Field(
        default=None,
        description="Optional list of specific paper IDs to download. If None, downloads all papers for the date.",
    )
    _validate_date = field_validator("date")(validate_date_format)


class GetPaperStatsRequest(BaseModel):
    days_back: int = Field(
        default=7,
        description="Number of days to look back for statistics. Use a larger value (e.g., 30, 365) for broader queries.",
    )


class GetTopPapersRequest(BaseModel):
    date: str = Field(
        description="Date to get papers for (format: YYYY-MM-DD).",
        examples=["2024-01-15"],
    )
    limit: int = Field(
        default=10,
        description="Maximum number of papers to return.",
    )
    _validate_date = field_validator("date")(validate_date_format)


class RankPapersRequest(BaseModel):
    date: str = Field(
        description="Single date to rank papers for (format: YYYY-MM-DD). Only one day at a time.",
        examples=["2024-01-15"],
    )

    @field_validator("date")
    @classmethod
    def validate_single_date(cls, v: str) -> str:
        """Reject time span patterns, then validate format."""
        # Check for spans first (more helpful error message)
        span_indicators = [" to ", " - ", "through", "between", ".."]
        v_lower = v.lower()
        for indicator in span_indicators:
            if indicator in v_lower:
                raise ValueError(
                    f"Time spans are not supported. Please request a single day's papers "
                    f"(e.g., '2024-01-15'). For multiple days, make separate requests."
                )
        # Then validate format
        validate_date_format(v)
        return v
