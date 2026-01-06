"""
Base agent class.

All agents inherit from this to get:
- Consistent LLM calling interface
- Prompt management (zero prompts in code)
- Tool integration
- Standard configuration
"""

from pathlib import Path
from typing import Any

from .prompts import PromptManager
from .tools import ToolRegistry


class Agent:
    """
    Base class for all DXTR agents.

    Key features:
    - Prompts loaded from filesystem (never hardcoded)
    - Consistent LLM calling interface
    - Built-in tool support
    - Configurable model and options
    """

    def __init__(
        self,
        name: str,
        model: str,
        prompts_dir: Path,
        default_options: dict | None = None,
    ):
        """
        Initialize agent.

        Args:
            name: Agent name (for logging/debugging)
            model: Ollama model name to use
            prompts_dir: Directory containing prompt markdown files
            default_options: Default options for LLM calls (temperature, num_ctx, etc.)
        """
        self.name = name
        self.model = model
        self.prompts = PromptManager(prompts_dir)
        self.tools = ToolRegistry()

        self.default_options = default_options or {
            "temperature": 0.3,
            "num_ctx": 16384,
        }

    def chat(
        self,
        messages: list[dict],
        prompt_name: str | None = None,
        use_tools: bool = False,
        stream: bool = False,
        **options,
    ) -> Any:
        """
        Chat with the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            prompt_name: Optional prompt to prepend as system message
            use_tools: Whether to make tools available to LLM
            stream: Whether to stream the response
            **options: Override default options (temperature, num_ctx, etc.)

        Returns:
            Response object or generator if streaming
        """
        # Load and prepend system prompt if specified
        if prompt_name:
            system_prompt = self.prompts.load(prompt_name)
            if not messages or messages[0].get("role") != "system":
                messages = [{"role": "system", "content": system_prompt}] + messages

        # Merge options
        llm_options = {**self.default_options, **options}

        # Prepare chat kwargs
        chat_kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": llm_options,
        }

        # Add tools if requested
        if use_tools and self.tools.get_definitions():
            chat_kwargs["tools"] = self.tools.get_definitions()

        return chat(**chat_kwargs)

    def chat_with_tool_calling(
        self,
        messages: list[dict],
        prompt_name: str | None = None,
        max_iterations: int = 5,
        **options,
    ) -> tuple[str, list[dict]]:
        """
        Chat with automatic tool calling loop.

        Handles the full tool calling flow:
        1. LLM generates response (possibly with tool calls)
        2. Execute tool calls
        3. Feed results back to LLM
        4. Repeat until no more tool calls (or max iterations)

        Args:
            messages: Initial message history
            prompt_name: Optional prompt to prepend
            max_iterations: Max tool calling iterations
            **options: LLM options

        Returns:
            tuple[str, list[dict]]: (final_response_text, full_message_history)
        """
        # Load system prompt if specified
        if prompt_name:
            system_prompt = self.prompts.load(prompt_name)
            if not messages or messages[0].get("role") != "system":
                messages = [{"role": "system", "content": system_prompt}] + messages

        iterations = 0
        while iterations < max_iterations:
            # Get response with tools
            response = self.chat(
                messages=messages, use_tools=True, stream=False, **options
            )

            # Check if there are tool calls
            if not response.message.tool_calls:
                # No tool calls, return final response
                return response.message.content, messages

            # Add assistant message with tool calls to history
            messages.append(response.message)

            # Execute each tool call
            for tool_call in response.message.tool_calls:
                function_name = tool_call.function.name
                function_args = tool_call.function.arguments

                try:
                    result = self.tools.call(function_name, **function_args)

                    # Format result for LLM
                    if isinstance(result, dict):
                        import json

                        result_str = json.dumps(result)
                    else:
                        result_str = str(result)

                    messages.append({"role": "tool", "content": result_str})
                except Exception as e:
                    # Add error message
                    messages.append(
                        {
                            "role": "tool",
                            "content": f"Error calling {function_name}: {str(e)}",
                        }
                    )

            iterations += 1

        # Max iterations reached, get final response
        final_response = self.chat(
            messages=messages, use_tools=False, stream=False, **options
        )

        return final_response.message.content, messages

    def __repr__(self) -> str:
        """String representation."""
        return f"Agent(name='{self.name}', model='{self.model}')"
