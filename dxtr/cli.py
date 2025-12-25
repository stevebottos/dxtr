#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from .agents.main.submodules import profile_creation
from .agents.main.agent import chat_with_agent


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
                            repo = f"{path_parts[idx+1]}/{path_parts[idx+2]}"
                            rel_file = "/".join(path_parts[idx+3:])

                            # Build keyword index
                            for keyword in keywords:
                                keyword_to_files[keyword.lower()].append(f"{repo}/{rel_file}")

                            # Store file summaries by repo
                            if summary:
                                repo_summaries[repo].append({
                                    "file": rel_file,
                                    "summary": summary,
                                    "keywords": keywords
                                })

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
                    keyword_to_files.items(),
                    key=lambda x: len(x[1]),
                    reverse=True
                )[:50]  # Top 50 most common keywords

                for keyword, files in sorted_keywords:
                    file_count = len(set(files))
                    context_parts.append(f"- **{keyword}**: {file_count} file(s)")

                # Repository Summaries
                context_parts.append("\n\n## Implementation Details by Repository\n")

                for repo, files in sorted(repo_summaries.items()):
                    context_parts.append(f"\n### {repo}\n")
                    for file_info in files[:10]:  # Limit to 10 files per repo to save space
                        context_parts.append(f"- `{file_info['file']}`: {file_info['summary']}")

                    if len(files) > 10:
                        context_parts.append(f"- ... and {len(files) - 10} more files")

        except Exception as e:
            context_parts.append(f"\n\n[Warning: Could not load GitHub analysis: {e}]")

    return "\n".join(context_parts)


def cmd_chat(args):
    """
    Start the main DXTR chat interface.

    On startup, calls get_user_profile to check/create the profile.
    Then enters the main chat loop.
    """
    print("=" * 80)
    print("DXTR - AI Research Assistant")
    print("=" * 80)
    print()

    # Check for user profile on startup
    profile_path = Path(".dxtr/dxtr_profile.md")

    if not profile_path.exists():
        print("No user profile found. Starting profile creation...\n")
        result = profile_creation.run()

        if result is None:
            print("\nProfile creation incomplete. Please create profile.md and restart DXTR.\n")
            return

        print("\n" + "=" * 80)
        print("Profile created! Starting chat...")
        print("=" * 80 + "\n")

    # Check profile exists before loading context
    if not profile_path.exists():
        print("Error: Profile still not found after creation.\n")
        return

    # Load user context (profile + GitHub analysis)
    print("[Loading user context...]")
    user_context = _load_user_context()
    print("[✓] Context loaded")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type '/help' for available commands.\n")

    # Main chat loop with user context prepended
    chat_history = [
        {
            "role": "system",
            "content": user_context
        }
    ]

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
                profile_creation.run()
                print("\nProfile updated. Reloading context...\n")

                # Reload context with new profile
                user_context = _load_user_context()
                chat_history = [
                    {
                        "role": "system",
                        "content": user_context
                    }
                ]
                print("[✓] Context reloaded. Continuing chat...\n")
                continue

            # Add user message to history
            chat_history.append({"role": "user", "content": user_input})

            # Get response from agent
            print("DXTR: ", end="", flush=True)
            response_text = ""

            for chunk in chat_with_agent(
                messages=chat_history,
                prompt_name="chat",
                stream=True,
            ):
                if hasattr(chunk.message, "content"):
                    content = chunk.message.content
                    response_text += content
                    print(content, end="", flush=True)

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
    parser_chat = subparsers.add_parser(
        "chat", help="Start the DXTR chat interface"
    )
    parser_chat.set_defaults(func=cmd_chat)

    args = parser.parse_args()

    # Default to chat if no command specified
    if args.command is None:
        cmd_chat(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
