"""Tools for the main chat agent."""

import asyncio
import json
from pathlib import Path

from pydantic_ai import Agent, RunContext

from agents.utils import model
from agents.subagents.profile_synthesis import github_utils


async def synthesize_profile(
    ctx: RunContext[None], seed_profile: str, github_summary: str
) -> str:
    """
    Create an enriched profile using the profile_synthesis agent.
    Takes the seed profile content and github summary as input.
    Saves result to .dxtr/profile.md.
    """
    context = f"""# Seed Profile

{seed_profile}

---

# GitHub Analysis

{github_summary}
"""

    result = await profile_synthesis.run(
        f"Create an enriched profile from the following:\n\n{context}"
    )

    # Save the profile
    profile_file = DXTR_DIR / "profile.md"
    profile_file.write_text(result.output)

    return f"Profile synthesized and saved to {profile_file}"
