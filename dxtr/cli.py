#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Import new configuration and agents
from dxtr.config_v2 import config
from dxtr.agents.main.agent import MainAgent
from dxtr.papers_etl import run_etl

def _load_user_context() -> str:
    """Load user profile content if it exists."""
    profile_path = config.paths.profile_file
    if profile_path.exists():
        return profile_path.read_text()
    return ""

def cmd_get_papers(args):
    """Get papers command - ETL pipeline for paper retrieval and processing"""
    run_etl(date=args.date, max_papers=args.max_papers)

import re

def parse_tool_tags(text: str) -> list[dict]:
    """
    Parse <tools>fn(arg='val'); ...</tools> format.
    Returns a list of tool call dicts: [{"tool": name, "parameters": {}}]
    """
    # 1. Extract content between <tools> and </tools>
    match = re.search(r"<tools>(.*?)</tools>", text, re.DOTALL)
    if not match:
        return []
    
    content = match.group(1).strip()
    # 2. Split by semicolon
    calls = [c.strip() for c in content.split(";") if c.strip()]
    
    results = []
    for call in calls:
        # Regex to match: fn_name(arg1='val1', arg2='val2')
        # This is a simple parser, might need improvement for complex values
        m = re.match(r"(\w+)\((.*)\)", call)
        if m:
            tool_name = m.group(1)
            params_str = m.group(2)
            
            # Parse parameters: key='val'
            params = {}
            # Matches: key='val' or key = 'val'
            p_matches = re.finditer(r"(\w+)\s*=\s*'([^']*)'", params_str)
            for pm in p_matches:
                params[pm.group(1)] = pm.group(2)
            
            results.append({"tool": tool_name, "parameters": params})
            
    return results

def _process_turn(agent, chat_history):
    """
    Process a single turn of conversation:
    - Generate response
    - Handle tool calls (recursively/iteratively)
    - Print output
    """
    while True:
        print("DXTR: ", end="", flush=True)
        
        response_generator = agent.chat(chat_history)
        full_response = ""
        
        # Consume generator
        for chunk in response_generator:
            print(chunk, end="", flush=True)
            full_response += chunk
        print() # Newline after response
        
        # Check for tool calls using new tag format
        tool_calls = parse_tool_tags(full_response)

        if tool_calls:
            all_results = []
            for tool_call in tool_calls:
                tool_name = tool_call.get("tool")
                tool_params = tool_call.get("parameters", {})
                
                print(f"\n[Calling Tool: {tool_name}]")
                
                result = None
                if hasattr(agent, tool_name):
                    try:
                        tool_func = getattr(agent, tool_name)
                        result = tool_func(**tool_params)
                    except Exception as e:
                        result = f"Error executing tool '{tool_name}': {e}"
                else:
                    result = f"Error: Tool '{tool_name}' not found."
                
                print(f"[Result: {str(result)[:100]}...]")
                
                # Context Reloading Hook
                if tool_name == "synthesize_profile" and "Profile synthesized" in str(result):
                    new_context = _load_user_context()
                    if new_context:
                        # Remove old profile system message if exists
                        chat_history[:] = [msg for msg in chat_history if not (msg.get("role") == "system" and "User Profile" in msg.get("content", ""))]
                        # Add new
                        chat_history.insert(0, {"role": "system", "content": f"User Profile:\n{new_context}"})
                        print("[System: User Profile Loaded]")
                
                all_results.append(f"Result of {tool_name}: {result}")
            
            # Add interactions to history
            chat_history.append({"role": "assistant", "content": full_response})
            # Combine all tool results into one response for simplicity in history
            combined_result = "\n".join(all_results)
            chat_history.append({"role": "tool", "content": combined_result})
            
            # Loop continues to let agent respond to tool result
            continue
        
        else:
            # Normal response
            chat_history.append({"role": "assistant", "content": full_response})
            break

def cmd_chat(args):
    """
    Start the main DXTR chat interface.
    """
    print("Initializing DXTR Agent (SGLang)...")
    try:
        agent = MainAgent()
    except Exception as e:
        print(f"Error initializing agent: {e}")
        print("Ensure SGLang server is running at localhost:30000")
        return

    # Initial context
    user_context = _load_user_context()
    
    print("\nDXTR - Research Assistant")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type '/help' for available commands.\n")

    # Chat history
    chat_history = []
    
    if user_context:
        # We silently load the profile. The agent will confirm it in the greeting.
        chat_history.append({"role": "system", "content": f"User Profile:\n{user_context}"})

    # Auto-Greeting / Pre-initialization
    # We inject a hidden user message to prompt the agent to start.
    print("Connecting to agent...\n")
    chat_history.append({"role": "user", "content": "Hello. Please check if my profile is loaded and briefly introduce yourself."})
    _process_turn(agent, chat_history)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye!\n")
                break

            if user_input == "/help":
                print("\nAvailable commands:")
                print("  /help     - Show this help message")
                print("  exit/quit - End the session\n")
                continue

            # Add user message
            chat_history.append({"role": "user", "content": user_input})

            # Process turn
            _process_turn(agent, chat_history)

        except KeyboardInterrupt:
            print("\n\nGoodbye!\n")
            break
        except Exception as e:
            print(f"\nError: {e}\n")

def main():
    parser = argparse.ArgumentParser(
        description="DXTR Agent - AI-powered research assistant"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # dxtr chat command
    parser_chat = subparsers.add_parser("chat", help="Start the DXTR chat interface")
    parser_chat.set_defaults(func=cmd_chat)

    # dxtr get-papers command
    parser_get_papers = subparsers.add_parser(
        "get-papers",
        help="Download and process daily papers (starts Docling service automatically)",
    )
    parser_get_papers.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date in YYYY-MM-DD format (default: today)",
    )
    parser_get_papers.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Maximum number of papers to process (default: all)",
    )
    parser_get_papers.set_defaults(func=cmd_get_papers)

    args = parser.parse_args()

    # Default to chat if no command specified
    if args.command is None:
        cmd_chat(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
