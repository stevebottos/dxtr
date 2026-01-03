#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from ollama import chat
from .config import config
from .agents.papers_helper.tools import paper_tools

# from .agents.profile_creator.tools import profile_tools
from .agents.deep_research.tools import deep_research_tools
from .papers_etl import run_etl


def _load_user_context():
    """
    Load user profile and GitHub analysis for context.

    Returns:
        str: Formatted context string with profile and GitHub analysis
    """
    context_parts = []

    # Load profile
    profile_path = Path(".dxtr/dxtr_profile.md")
    if profile_path.exists():
        profile_content = profile_path.read_text()
        context_parts.append("# User Profile\n\n" + profile_content)

    # Load and parse GitHub summary
    github_summary_path = Path(".dxtr/github_summary.json")
    if github_summary_path.exists():
        try:
            github_summary = json.loads(github_summary_path.read_text())

            # Extract all keywords and create technology index
            keyword_to_files = defaultdict(list)
            repo_summaries = defaultdict(list)

            for file_path, analysis_json in github_summary.items():
                try:
                    analysis = json.loads(analysis_json)
                    keywords = analysis.get("keywords", [])
                    summary = analysis.get("summary", "")

                    # Extract repo name from path
                    path_parts = Path(file_path).parts
                    if "repos" in path_parts:
                        idx = path_parts.index("repos")
                        if idx + 2 < len(path_parts):
                            repo = f"{path_parts[idx + 1]}/{path_parts[idx + 2]}"
                            rel_file = "/".join(path_parts[idx + 3 :])

                            # Build keyword index
                            for keyword in keywords:
                                keyword_to_files[keyword.lower()].append(
                                    f"{repo}/{rel_file}"
                                )

                            # Store file summaries by repo
                            if summary:
                                repo_summaries[repo].append(
                                    {
                                        "file": rel_file,
                                        "summary": summary,
                                        "keywords": keywords,
                                    }
                                )

                except json.JSONDecodeError:
                    continue

            # Format GitHub analysis context
            if keyword_to_files or repo_summaries:
                context_parts.append("\n\n---\n\n# GitHub Code Analysis")

                # Technology Index
                context_parts.append("\n## Technologies & Libraries Used")
                context_parts.append("\n(Keyword → Files where it appears)\n")

                # Sort keywords by frequency
                sorted_keywords = sorted(
                    keyword_to_files.items(), key=lambda x: len(x[1]), reverse=True
                )[:50]  # Top 50 most common keywords

                for keyword, files in sorted_keywords:
                    file_count = len(set(files))
                    context_parts.append(f"- **{keyword}**: {file_count} file(s)")

                # Repository Summaries
                context_parts.append("\n\n## Implementation Details by Repository\n")

                for repo, files in sorted(repo_summaries.items()):
                    context_parts.append(f"\n### {repo}\n")
                    for file_info in files[
                        :10
                    ]:  # Limit to 10 files per repo to save space
                        context_parts.append(
                            f"- `{file_info['file']}`: {file_info['summary']}"
                        )

                    if len(files) > 10:
                        context_parts.append(f"- ... and {len(files) - 10} more files")

        except Exception as e:
            context_parts.append(f"\n\n[Warning: Could not load GitHub analysis: {e}]")

    return "\n".join(context_parts)


def cmd_get_papers(args):
    """Get papers command - ETL pipeline for paper retrieval and processing"""
    run_etl(date=args.date, max_papers=args.max_papers)


def cmd_chat(args):
    """
    Start the main DXTR chat interface.
    """
    profile_path = Path(".dxtr/dxtr_profile.md")

    if not profile_path.exists():
        print("\n" + "=" * 80)
        print("NO PROFILE FOUND")
        print("=" * 80)
        print("\nTo get started:")
        print("1. Create a file named 'profile.md' with your information")
        print("2. Include your GitHub profile URL")
        print("3. Ask me to create your profile!\n")
        print("Or type '/profile' to create your profile now.\n")

    # Load user context if profile exists
    user_context = ""
    if profile_path.exists():
        print("[Loading user context...]")
        user_context = _load_user_context()
        print("[✓] Context loaded")

    print("Type 'exit' or 'quit' to end the session.")
    print("Type '/help' for available commands.\n")

    # Main chat loop with user context prepended
    chat_history = [{"role": "system", "content": user_context}]

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye!\n")
                break

            if user_input == "/help":
                print("\nAvailable commands:")
                print("  /help     - Show this help message")
                print("  /profile  - Recreate your profile")
                print("  exit/quit - End the session\n")
                continue

            if user_input == "/profile":
                print("\nStarting profile recreation...\n")
                result = profile_tools.create_profile()

                if result.get("success"):
                    print("\nProfile updated. Reloading context...\n")
                    user_context = _load_user_context()
                    chat_history = [{"role": "system", "content": user_context}]
                    print("[✓] Context reloaded. Continuing chat...\n")
                else:
                    print(f"\nError: {result.get('error')}\n")
                continue

            # Add user message to history
            chat_history.append({"role": "user", "content": user_input})

            # Available tools
            tools = [
                paper_tools.TOOL_DEFINITION,
                profile_tools.TOOL_DEFINITION,
                deep_research_tools.TOOL_DEFINITION,
            ]
            available_functions = {
                "rank_papers": paper_tools.rank_papers,
                "create_profile": profile_tools.create_profile,
                "deep_research": deep_research_tools.deep_research,
            }

            # Get response from agent with tools (streaming)
            main_config = config.get_model_config("main")
            print("DXTR: ", end="", flush=True)

            response_text = ""
            tool_calls = None

            response_stream = chat(
                model=main_config.name,
                messages=chat_history,
                tools=tools,
                stream=True,
                options={
                    "temperature": main_config.temperature,
                    "num_ctx": main_config.context_window,
                },
            )

            # Stream response and check for tool calls
            for chunk in response_stream:
                if hasattr(chunk.message, "content") and chunk.message.content:
                    content = chunk.message.content
                    response_text += content
                    print(content, end="", flush=True)

                # Tool calls come in the final chunk
                if hasattr(chunk.message, "tool_calls") and chunk.message.tool_calls:
                    tool_calls = chunk.message.tool_calls

            # Handle tool calls if any
            if tool_calls:
                print("\n")  # New line after initial response

                # Add message to history
                chat_history.append(
                    {
                        "role": "assistant",
                        "content": response_text,
                        "tool_calls": tool_calls,
                    }
                )

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = tool_call.function.arguments

                    if function_name in available_functions:
                        print(f"[Calling {function_name}...]")
                        function_result = available_functions[function_name](
                            **function_args
                        )

                        # Format tool result for better model understanding
                        if function_result.get("success"):
                            if function_name == "rank_papers":
                                tool_content = f"Successfully ranked papers for {function_result['date']}:\n\n{function_result['ranking']}"
                            elif function_name == "create_profile":
                                tool_content = f"Successfully created profile at {function_result['profile_path']}. User context has been updated."
                                # Reload context after profile creation
                                user_context = _load_user_context()
                                chat_history[0] = {
                                    "role": "system",
                                    "content": user_context,
                                }
                            elif function_name == "deep_research":
                                tool_content = f"Deep research answer for paper {function_result['paper_id']}:\n\n{function_result['answer']}"
                            else:
                                tool_content = json.dumps(function_result)
                        else:
                            tool_content = f"Error: {function_result.get('error', 'Unknown error')}"

                        chat_history.append({"role": "tool", "content": tool_content})

                # Tool output already streamed - no need for main agent to synthesize
                print("\n[Tool complete - added to context]\n")

            del main_config

            print("\n")

            # Add assistant response to history
            chat_history.append({"role": "assistant", "content": response_text})

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
