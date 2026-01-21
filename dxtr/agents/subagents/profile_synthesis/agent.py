from pathlib import Path

from pydantic_ai import Agent

from dxtr import load_system_prompt, profile_synthesizer, data_models


agent = Agent(
    profile_synthesizer,
    system_prompt=load_system_prompt(Path(__file__).parent / "system.md"),
    # deps_type=data_models.ProfileSynthesisRequest,
)
