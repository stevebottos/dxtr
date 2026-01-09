"""
Main Agent - Lightweight chat agent that coordinates sub-agents via tools.

Responsibilities:
- Handle chat functionality with minimal context
- Offload context-heavy operations to specialized agents
- All sub-agents return results to main when work is done
"""

import json
import requests
from pathlib import Path
from typing import Generator

from dxtr.agents.base import BaseAgent
from dxtr.config_v2 import config
from dxtr.agents.github_summarize.agent import Agent as GithubSummarizeAgent
from dxtr.agents.profile_synthesize.agent import Agent as ProfileSynthesizeAgent


# Tool Definitions - these map to methods on MainAgent
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the content of a file. Use this to read the user's seed profile.md.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file to read"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_github",
            "description": "Analyze GitHub repos from the seed profile. Extracts GitHub URL from profile, clones pinned repos, and creates a summary. Saves result to .dxtr/github_summary.json.",
            "parameters": {
                "type": "object",
                "properties": {
                    "profile_path": {"type": "string", "description": "Path to the seed profile file containing GitHub URL"}
                },
                "required": ["profile_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "synthesize_profile",
            "description": "Synthesize the final user profile from available artifacts in .dxtr/ directory. Reads github_summary.json and other artifacts, then creates a comprehensive profile. Saves result to .dxtr/profile.md.",
            "parameters": {
                "type": "object",
                "properties": {
                    "seed_profile_path": {"type": "string", "description": "Path to the original seed profile.md provided by user"}
                },
                "required": ["seed_profile_path"]
            }
        }
    }
]

class MainAgent(BaseAgent):
    """Lightweight chat agent that coordinates sub-agents via tools."""

    def __init__(self):
        """Initialize main agent."""
        super().__init__()
        self.system_prompt = self.load_system_prompt(
            Path(__file__).parent / "system.md"
        )
        # Track seed profile path for the session
        self.seed_profile_path: str | None = None

    # --- Tool Methods (called by CLI via getattr) ---

    def read_file(self, file_path: str) -> str:
        """Read content from a file."""
        try:
            path = Path(file_path).expanduser().resolve()
            if not path.exists():
                return f"Error: File not found: {file_path}"
            content = path.read_text()
            # Store seed profile path if this looks like a profile
            if "profile" in file_path.lower() or file_path.endswith(".md"):
                self.seed_profile_path = str(path)
            return content
        except Exception as e:
            return f"Error reading file: {e}"

    def summarize_github(self, profile_path: str) -> str:
        """Run GitHub summarize agent on the profile."""
        try:
            path = Path(profile_path).expanduser().resolve()
            if not path.exists():
                return f"Error: Profile not found: {profile_path}"

            agent = GithubSummarizeAgent()
            result = agent.run(profile_path=path)

            if result:
                return f"GitHub summary complete. Analyzed {len(result)} files. Saved to .dxtr/github_summary.json"
            else:
                return "No GitHub URL found in profile or no repos to analyze."
        except Exception as e:
            return f"Error running GitHub summarize: {e}"

    def synthesize_profile(self, seed_profile_path: str) -> str:
        """Run profile synthesis agent to create final profile from artifacts."""
        try:
            path = Path(seed_profile_path).expanduser().resolve()
            if not path.exists():
                return f"Error: Seed profile not found: {seed_profile_path}"

            agent = ProfileSynthesizeAgent()
            result = agent.run(seed_profile_path=path)

            if result:
                return f"Profile synthesized successfully. Saved to {config.paths.profile_file}"
            else:
                return "Error: Profile synthesis returned empty result."
        except Exception as e:
            return f"Error running profile synthesis: {e}"

    # --- Chat Method ---

    def chat(self, messages: list[dict], stream: bool = True) -> Generator[dict, None, None]:
        """
        Chat with the agent using native tool calling.

        Args:
            messages: List of message dicts
            stream: Whether to stream response (always True for now)

        Yields:
            Dict with either {"type": "content", "data": str} or {"type": "tool_calls", "data": list}
        """
        # Refresh state
        self.state.check_state()

        # Inject global state into system prompt
        state_str = f"Global State: {self.state}"
        full_system_prompt = f"{self.system_prompt}\n\n{state_str}"

        # Construct messages payload for OpenAI-compatible API
        api_messages = [{"role": "system", "content": full_system_prompt}]

        for msg in messages:
            role = msg["role"]
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")

            if role == "tool":
                # Native tool response format
                api_messages.append({
                    "role": "tool",
                    "content": content,
                    "tool_call_id": tool_call_id
                })
            elif role == "assistant" and tool_calls:
                # Assistant message with tool calls
                api_messages.append({
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": tool_calls
                })
            elif role == "system" and "User Profile" in (content or ""):
                api_messages.append({"role": "system", "content": content})
            else:
                api_messages.append({"role": role, "content": content})

        # Use requests to stream from SGLang server
        url = f"{config.sglang.base_url}/chat/completions"
        payload = {
            "model": "default",
            "messages": api_messages,
            "tools": TOOLS,
            "temperature": 0.3,
            "max_tokens": 1000,
            "stream": True
        }

        try:
            accumulated_tool_calls = {}

            with requests.post(url, json=payload, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                delta = data.get("choices", [{}])[0].get("delta", {})

                                # Handle content
                                content = delta.get("content", "")
                                if content:
                                    yield {"type": "content", "data": content}

                                # Handle tool calls (streamed incrementally)
                                # Note: delta may have tool_calls=None, so use 'or []'
                                tool_calls = delta.get("tool_calls") or []
                                for tc in tool_calls:
                                    idx = tc.get("index", 0)
                                    if idx not in accumulated_tool_calls:
                                        accumulated_tool_calls[idx] = {
                                            "id": tc.get("id", ""),
                                            "type": "function",
                                            "function": {"name": "", "arguments": ""}
                                        }

                                    if tc.get("id"):
                                        accumulated_tool_calls[idx]["id"] = tc["id"]

                                    func = tc.get("function", {})
                                    if func.get("name"):
                                        accumulated_tool_calls[idx]["function"]["name"] += func["name"]
                                    if func.get("arguments"):
                                        accumulated_tool_calls[idx]["function"]["arguments"] += func["arguments"]

                            except json.JSONDecodeError:
                                continue

                # Yield accumulated tool calls at the end if any
                if accumulated_tool_calls:
                    yield {"type": "tool_calls", "data": list(accumulated_tool_calls.values())}

        except Exception as e:
            yield {"type": "content", "data": f"Error generating response: {e}"}