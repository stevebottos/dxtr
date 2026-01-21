"""
This module just defines requests/responses relating to agent interfaces, not necessarily
tool interfaces.
"""

from pydantic import BaseModel, Field


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
    base_url: str = Field(
        description="GitHub profile BASE URL only, format: https://github.com/<username>. "
        "Must NOT be a repository URL (e.g. https://github.com/user/repo).",
        examples=["https://github.com/stevebottos", "https://github.com/anthropics"],
    )


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
