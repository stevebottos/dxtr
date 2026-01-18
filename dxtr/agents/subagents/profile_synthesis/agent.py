from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from dxtr import DXTR_DIR, load_system_prompt, profile_synthesizer


class ProfileSynthesisDeps(BaseModel):
    seed_profile: str = Field(description="User's self-description from their profile file")
    github_summary: str = Field(description="JSON summary of analyzed GitHub repositories")


agent = Agent(
    profile_synthesizer,
    system_prompt=load_system_prompt(Path(__file__).parent / "system.md"),
    deps_type=ProfileSynthesisDeps,
)
