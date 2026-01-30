from pathlib import Path

from pydantic_ai import Agent

from dxtr import load_system_prompt, profile_synthesizer, get_session_state


SYSTEM_PROMPT_BASE = load_system_prompt(Path(__file__).parent / "system.md")

agent = Agent(profile_synthesizer)


@agent.system_prompt
def build_system_prompt() -> str:
    """Build system prompt with existing profile (if any) for reference."""
    state = get_session_state()

    if state.profile_content:
        profile_section = f"""
# Existing Profile (for reference)
The user already has a profile. Use this as context when creating the updated version.

{state.profile_content}
"""
    else:
        profile_section = """
# Existing Profile
None - this is a new profile.
"""

    return SYSTEM_PROMPT_BASE + profile_section
