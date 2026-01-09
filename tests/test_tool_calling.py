import json
import pytest
import requests
import re
from pathlib import Path
from dxtr.config_v2 import config

# --- Test Configuration ---
NUM_RUNS = 10  # Reduced for faster iteration during rework
API_URL = f"{config.sglang.base_url}/chat/completions"

# --- Extraction Logic (More robust version for Qwen) ---
def parse_tool_tags(text: str) -> list[dict]:
    """Parse <tools>fn(arg='val'); ...</tools> format."""
    # Qwen doesn't usually output <think> but we keep it for compatibility
    clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    match = re.search(r"<tools>(.*?)</tools>", clean_text, re.DOTALL)
    if not match:
        return []
    
    content = match.group(1).strip()
    calls = [c.strip() for c in content.split(";") if c.strip()]
    
    results = []
    for call in calls:
        m = re.match(r"(\w+)\((.*)\)", call)
        if m:
            tool_name = m.group(1)
            params_str = m.group(2)
            params = {}
            # Support both single and double quotes for robustness with smaller models
            p_matches = re.finditer(r"(\w+)\s*=\s*['\"]([^'\"]*)['\"]", params_str)
            for pm in p_matches:
                params[pm.group(1)] = pm.group(2)
            results.append({"tool": tool_name, "parameters": params})
    return results

def get_test_system_prompt():
    """Self-contained system prompt for tool calling tests."""
    return """You are DXTR, a research assistant.
You MUST use tools to fulfill requests when necessary.

### TOOL PROTOCOL
To call a tool, you MUST use this exact format:
<tools>tool_name(parameter='value')</tools>

### RULES
1. ALWAYS use single quotes for string values inside the tool call.
2. DO NOT include any text before or after the <tools> tags if you are calling a tool.
3. If you need a file path, use the one provided by the user.

### AVAILABLE TOOLS
- read_file(file_path: str): Read content from a local file.
- summarize_github(profile_path: str): Process and summarize GitHub activity.
- synthesize_profile(profile_path: str): Create a research profile.
"""

def call_llm(messages):
    """Sync call to LLM for testing."""
    payload = {
        "model": "default",
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1000,
        "stream": False 
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        pytest.fail(f"LLM call failed: {e}")

# --- Test Cases ---

TEST_SCENARIOS = [
    {
        "name": "read_profile",
        "user_input": "My profile is at /home/steve/repos/dxtr-cli/profile.md",
        "expected_tool": "synthesize_profile",
        "expected_params": ["profile_path"]
    },
    {
        "name": "summarize_github",
        "setup_history": [
            {"role": "user", "content": "My profile is at /home/steve/repos/dxtr-cli/profile.md"},
            {"role": "assistant", "content": "<tools>read_file(file_path='/home/steve/repos/dxtr-cli/profile.md')</tools>"},
            {"role": "tool", "content": "Result of read_file: Profile content with GitHub URL: https://github.com/steve"},
        ],
        "user_input": "Great, now summarize my github and synthesize my profile.",
        "expected_tool": "summarize_github",
        "expected_params": ["profile_path"]
    }
]

@pytest.mark.parametrize("run_id", range(NUM_RUNS))
@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_tool_calling_consistency(run_id, scenario):
    """Test that Qwen3B consistently calls the correct tool with valid tags."""
    system_prompt = get_test_system_prompt()
    
    messages = [{"role": "system", "content": system_prompt}]
    
    if "setup_history" in scenario:
        messages.extend(scenario["setup_history"])
    
    messages.append({"role": "user", "content": scenario["user_input"]})
    
    # Execution
    output = call_llm(messages)

    # Verification
    tool_calls = parse_tool_tags(output)
    
    assert len(tool_calls) > 0, f"Run {run_id}: No tool calls extracted from output: {output}"
    
    # Validate the first tool call
    tool_call = tool_calls[0]
    assert tool_call["tool"] == scenario["expected_tool"], f"Run {run_id}: Expected {scenario['expected_tool']}, got {tool_call['tool']}. Output: {output}"
    
    params = tool_call.get("parameters", {})
    for p in scenario["expected_params"]:
        assert p in params, f"Run {run_id}: Missing parameter '{p}' in tool call. Params: {params}"
        assert params[p], f"Run {run_id}: Parameter '{p}' is empty"
