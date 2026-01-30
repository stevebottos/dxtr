"""
This module just defines requests/responses relating to agent interfaces, not necessarily
tool interfaces.
"""

import re
from typing import List

from pydantic import BaseModel, Field, field_validator

# Pattern for safe IDs: UUIDs, alphanumeric, underscores, hyphens
# Prevents path traversal attacks (e.g., ../../../etc)
SAFE_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')


# These fields are set explicitly via tha chat interface
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


class MasterResponse(BaseModel):
    answer: str


# These fields are set implicitly by the LLM, so we have descriptors for them
class ProfileSynthesisRequest(MasterRequest):
    with_github_summary: bool = Field(
        description="""Whether or not to pull a previously computed github summary from the user's profile for synthesis.""",
    )


class GithubSummarizerRequest(MasterRequest):
    # base_url: str = Field(
    #     description="GitHub profile BASE URL only, format: https://github.com/<username>. "
    #     "Must NOT be a repository URL (e.g. https://github.com/user/repo).",
    #     examples=["https://github.com/stevebottos", "https://github.com/anthropics"],
    # )
    repo_urls: List[str] = Field(
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
    def ensure_git_suffix(cls, urls: List[str]) -> List[str]:
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


class SessionState(BaseModel):
    """User state loaded at the start of each turn.

    Injected into system prompt so the agent knows what artifacts exist
    without needing to call a tool.

    TODO: Migrate from GCS to database for faster reads.
    """
    has_synthesized_profile: bool = False
    has_github_summary: bool = False
    profile_content: str | None = None  # Full profile text, injected into system prompt
