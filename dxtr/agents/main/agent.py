"""
Main Agent - Handles chat and coordinates submodules

The main agent has:
- Main chat mode (prompts/chat.md)
- Submodules with their own prompts (e.g., profile_creation)
"""

from pathlib import Path
from ollama import chat

# Model to use for this agent
# MODEL = "mistral-nemo"
# MODEL = "nemotron-3-nano"
MODEL = "gemma3:12b"


def _load_system_prompt(prompt_name="chat"):
    """
    Load a system prompt from the prompts directory.

    Args:
        prompt_name: Name of the prompt file (without .md extension)
                    Options: "chat", "profile_creation", etc.

    Returns:
        str: The system prompt content
    """
    prompt_path = Path(__file__).parent / "prompts" / f"{prompt_name}.md"
    return prompt_path.read_text().strip()


def chat_with_agent(messages, prompt_name="chat", stream=True, **options):
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
    # Default options
    default_options = {
        "temperature": 0.3,
        "num_ctx": 4096 * 4,  # 16384 context window
    }
    default_options.update(options)

    # Prepend system message if not already present
    if not messages or messages[0].get("role") != "system":
        system_prompt = _load_system_prompt(prompt_name)
        messages = [{"role": "system", "content": system_prompt}] + messages

    return chat(
        model=MODEL,
        messages=messages,
        stream=stream,
        options=default_options,
    )
