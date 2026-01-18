from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from dxtr import DXTR_DIR, master, load_system_prompt, get_model_settings
from dxtr.agents.subagents import github_summarizer
from dxtr.agents.subagents import profile_synthesis


SYSTEM_PROMPT = load_system_prompt(Path(__file__).parent / "system.md")

agent = Agent(master, system_prompt=SYSTEM_PROMPT)


class GitHubProfileRequest(BaseModel):
    base_url: str = Field(
        description="GitHub profile BASE URL only, format: https://github.com/<username>. "
        "Must NOT be a repository URL (e.g. https://github.com/user/repo).",
        examples=["https://github.com/stevebottos", "https://github.com/anthropics"],
    )


class FileReadRequest(BaseModel):
    file_path: str = Field(
        description="Absolute or relative path to a file to read.",
        examples=["~/.profile.md", "/home/user/documents/resume.txt"],
    )


class ProfileSynthesisRequest(BaseModel):
    seed_profile: str = Field(
        description="The user's self-description content (from their profile file).",
    )
    github_summary: str = Field(
        description="The JSON summary from GitHub analysis.",
    )


@agent.tool_plain
async def call_github_summarizer(request: GitHubProfileRequest) -> str:
    """Analyze a user's GitHub profile: clone pinned repos, read code, generate summary."""
    print("Call: Github Summary Agent")
    result = await github_summarizer.agent.run(
        "Analyze the user's GitHub profile.",
        deps=request.base_url,
        model_settings=get_model_settings(),
    )
    return result.output


@agent.tool_plain
async def read_file(request: FileReadRequest) -> str:
    """Read content from a file."""
    try:
        path = Path(request.file_path).expanduser().resolve()
        if not path.exists():
            return f"Error: File not found: {request.file_path}"
        return path.read_text()
    except Exception as e:
        return f"Error reading file: {e}"


@agent.tool_plain
async def call_profile_synthesizer(request: ProfileSynthesisRequest) -> str:
    """Synthesize an enriched user profile from seed profile and GitHub analysis.
    If the user has provided a github profile, you need to handle that first."""
    print("Call: Profile Synthesis Agent")
    deps = profile_synthesis.ProfileSynthesisDeps(
        seed_profile=request.seed_profile,
        github_summary=request.github_summary,
    )
    result = await profile_synthesis.agent.run(
        f"Create an enriched profile.\n\nSeed Profile:\n{request.seed_profile}\n\nGitHub Analysis:\n{request.github_summary}",
        deps=deps,
        model_settings=get_model_settings(),
    )

    # Save synthesized profile
    profile_file = DXTR_DIR / "synthesized_profile.md"
    profile_file.write_text(result.output)
    print(f"  Saved to {profile_file}")

    return result.output
