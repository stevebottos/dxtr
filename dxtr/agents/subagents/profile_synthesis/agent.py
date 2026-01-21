from pathlib import Path

from pydantic_ai import Agent

from dxtr import load_system_prompt, profile_synthesizer


agent = Agent(
    profile_synthesizer,
    system_prompt=load_system_prompt(Path(__file__).parent / "system.md"),
)


# No tools yet... None are really necessary. I know that having this file is pointless.
