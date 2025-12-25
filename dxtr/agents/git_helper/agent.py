"""
Git Helper Agent

Handles GitHub repository analysis:
1. Scrapes pinned repos from GitHub profiles
2. Clones repositories
3. Analyzes Python code using LLM
4. Returns github_summary.json data
"""

import json
from pathlib import Path
from ollama import chat
from .tools import git_tools
from . import analyzer

# Model for code analysis
MODEL = "qwen2.5-coder"


def _load_system_prompt(prompt_name: str) -> str:
    """
    Load a system prompt from the prompts directory.

    Args:
        prompt_name: Name of the prompt file (without .md extension)

    Returns:
        str: The system prompt content
    """
    prompt_path = Path(__file__).parent / "prompts" / f"{prompt_name}.md"
    return prompt_path.read_text()


def run(github_url: str) -> dict[str, str]:
    """
    Analyze GitHub pinned repositories.

    This is the main entry point for the git_helper agent.

    Args:
        github_url: GitHub profile URL

    Returns:
        Dict mapping file paths to JSON analysis strings
        Format: {"/path/to/file.py": '{"keywords": [...], "summary": "..."}', ...}
    """
    print(f"\n[Git Helper Agent: Analyzing {github_url}]")

    # Check if it's a profile URL
    if not git_tools.is_profile_url(github_url):
        print(f"  [Error: Not a GitHub profile URL]")
        return {}

    # Fetch profile HTML
    html = git_tools.fetch_profile_html(github_url)
    if not html:
        print("  [Failed to fetch profile HTML]")
        return {}

    # Extract pinned repos
    pinned_repos = git_tools.extract_pinned_repos(html)

    # Filter out dxtr-cli
    pinned_repos = [repo for repo in pinned_repos if not repo.endswith('/dxtr-cli')]

    if not pinned_repos:
        print("  [No pinned repositories found]")
        return {}

    print(f"  [Found {len(pinned_repos)} pinned repository(ies)]")

    # Clone repos
    print(f"\n[Cloning repositories...]")
    clone_results = []
    for repo_url in pinned_repos:
        result = git_tools.clone_repo(repo_url)
        clone_results.append(result)

        if result["success"]:
            status = "✓ cached" if "cached" in result["message"].lower() else "✓ cloned"
            print(f"  [{status}] {result['owner']}/{result['repo']}")
        else:
            print(f"  [✗ failed] {result['url']}: {result['message']}")

    # Analyze repos
    successful = [r for r in clone_results if r["success"]]
    if not successful:
        print("  [No repositories to analyze]")
        return {}

    print(f"\n[Analyzing {len(successful)} repository(ies)...]")

    github_summary = {}

    # Load module analysis prompt
    module_analysis_prompt = _load_system_prompt("module_analysis")

    for result in successful:
        repo_path = Path(result["path"])
        print(f"  [Analyzing {result['owner']}/{result['repo']}...]")

        # Find Python files
        python_files = analyzer.find_python_files(repo_path)

        if not python_files:
            print(f"    [No Python files found]")
            continue

        print(f"    [Found {len(python_files)} Python file(s)]")

        # Analyze each file
        for idx, py_file in enumerate(python_files, 1):
            rel_path = py_file.relative_to(repo_path)
            print(f"    [{idx}/{len(python_files)}] {rel_path}", end=" ", flush=True)

            try:
                source_code = py_file.read_text(encoding='utf-8')
                file_size_kb = len(source_code) / 1024

                # LLM analysis
                response = chat(
                    model=MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": module_analysis_prompt
                        },
                        {
                            "role": "user",
                            "content": source_code
                        }
                    ],
                    options={
                        "temperature": 0.3,
                        "num_ctx": 16384,
                    },
                    format=analyzer.MODULE_ANALYSIS_SCHEMA
                )

                # Store the JSON string
                github_summary[str(py_file)] = response.message.content

                # Show token counts
                prompt_tokens = response.get("prompt_eval_count", 0)
                completion_tokens = response.get("eval_count", 0)
                print(f"({file_size_kb:.1f}KB, {prompt_tokens + completion_tokens} tokens)")

            except Exception as e:
                print(f"[ERROR: {str(e)}]")
                continue

    print(f"\n  [✓] Git Helper Agent: Analyzed {len(github_summary)} file(s) total")

    return github_summary
