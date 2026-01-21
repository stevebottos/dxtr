"""
This module just defines requests/responses relating to agent interfaces, not necessarily
tool interfaces.
"""

from typing import List

from pydantic import BaseModel, Field, field_validator


# These fields are set explicitly via tha chat interface
class MasterRequest(BaseModel):
    user_id: str
    session_id: str
    query: str


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


# TODO: This is broken since we're in GCS now
class FileReadRequest(BaseModel):
    file_path: str = Field(
        description="Absolute or relative path to a file to read.",
        examples=["~/.profile.md", "/home/user/documents/resume.txt"],
    )
