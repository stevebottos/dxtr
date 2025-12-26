"""
Main Agent - Handles chat and coordinates submodules

The main agent has:
- Main chat mode (prompts/chat.md)
- Submodules with their own prompts (e.g., profile_creation)
"""

from pathlib import Path
from dxtr.base import Agent
from dxtr.config import config


class MainAgent(Agent):
    """Main agent for chat and coordination."""

    def __init__(self):
        """Initialize main agent."""
        model_config = config.get_model_config("main")
        super().__init__(
            name="main",
            model=model_config.name,
            prompts_dir=Path(__file__).parent / "prompts",
            default_options={
                "temperature": model_config.temperature,
                "num_ctx": model_config.context_window,
            }
        )

    def chat_with_agent(self, messages, prompt_name="chat", stream=True, **options):
        """
        Chat with the main agent using the specified prompt.

        Args:
            messages: List of message dicts with 'role' and 'content'
            prompt_name: Which system prompt to use ("chat", "profile_creation", etc.)
            stream: Whether to stream the response (default: True)
            **options: Additional options to pass to the chat function

        Returns:
            Generator of chat response chunks if stream=True, otherwise the full response
        """
        return self.chat(
            messages=messages,
            prompt_name=prompt_name,
            stream=stream,
            use_tools=False,
            **options
        )


# Global instance for backward compatibility
_agent = MainAgent()


def chat_with_agent(messages, prompt_name="chat", stream=True, **options):
    """
    Chat with the main agent using the specified prompt.

    This is a convenience function that delegates to the agent instance.

    Args:
        messages: List of message dicts with 'role' and 'content'
        prompt_name: Which system prompt to use ("chat", "profile_creation", etc.)
        stream: Whether to stream the response (default: True)
        **options: Additional options to pass to the chat function

    Returns:
        Generator of chat response chunks if stream=True, otherwise the full response
    """
    return _agent.chat_with_agent(messages, prompt_name, stream, **options)
