#!/usr/bin/env python3
import argparse
from pathlib import Path
from .agents.main.submodules import profile_creation
from .agents.main.agent import chat_with_agent


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
    else:
        print(f"User profile loaded from {profile_path}\n")
        print("Type 'exit' or 'quit' to end the session.")
        print("Type '/help' for available commands.\n")

    # Main chat loop
    chat_history = []

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
                print("\nProfile updated. Continuing chat...\n")
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
