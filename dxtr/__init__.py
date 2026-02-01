from dotenv import load_dotenv

load_dotenv()

from pathlib import Path


def load_system_prompt(file_path: Path) -> str:
    """Load a system prompt from a markdown file."""
    return file_path.read_text().strip()


async def run_agent(agent, query: str, deps, **kwargs):
    """Run an agent with a prompt.

    Args:
        agent: The pydantic-ai Agent to run
        query: The prompt/query string
        deps: Dependencies for the agent (must match agent's deps_type, or None if no deps_type)
        **kwargs: Additional args passed to agent.run() (e.g., model_settings)
    """
    return await agent.run(query, deps=deps, **kwargs)
