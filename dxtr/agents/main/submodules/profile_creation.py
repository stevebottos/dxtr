"""
Profile Creation Submodule

Handles automatic profile creation/enrichment:
1. Reads seed profile.md
2. Extracts GitHub profile URL
3. Scrapes pinned repos, clones them, analyzes code
4. Generates github_summary.json
5. Creates enriched .dxtr/dxtr_profile.md
"""

import json
import re
from pathlib import Path
from ollama import chat
from ..agent import MODEL
from ..tools import git_tools
from . import github_explorer


def _extract_github_profile_url(profile_content: str) -> str | None:
    """
    Extract GitHub profile URL from profile content.

    Args:
        profile_content: Raw profile.md content

    Returns:
        GitHub profile URL or None if not found
    """
    # Find all GitHub URLs
    url_pattern = r'https?://github\.com/[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, profile_content)

    # Filter to profile URLs only
    for url in urls:
        if git_tools.is_profile_url(url):
            return url

    return None


def _analyze_github_repos(github_url: str) -> dict[str, str]:
    """
    Scrape, clone, and analyze GitHub pinned repositories.

    Args:
        github_url: GitHub profile URL

    Returns:
        Dict mapping file paths to JSON analysis strings
    """
    print(f"\n[Fetching GitHub profile: {github_url}]")

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

    for result in successful:
        repo_path = Path(result["path"])
        print(f"  [Analyzing {result['owner']}/{result['repo']}...]")

        # Find Python files
        python_files = github_explorer.find_python_files(repo_path)

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
                    model=github_explorer.ANALYSIS_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": """Assess this Python module for relevant experience, technologies, techniques, and implementations.

Focus on:
- Specific libraries/frameworks used (e.g., "torch.nn", "transformers", "scann")
- Modern techniques present or absent (e.g., "ROPE", "flash attention", "KV caching")
- Implementation patterns (e.g., "custom attention", "CUDA kernels", "NumPy-based")
- Architectural approaches (e.g., "encoder-decoder", "transformer", "CNN")

Output:
1. Comprehensive list of keywords (technical terms, libraries, techniques)
2. Brief module-level summary (2-3 sentences)"""
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
                    format=github_explorer.MODULE_ANALYSIS_SCHEMA
                )

                # Store the JSON string (matching embed_github_repo.py)
                github_summary[str(py_file)] = response.message.content

                # Show token counts
                prompt_tokens = response.get("prompt_eval_count", 0)
                completion_tokens = response.get("eval_count", 0)
                print(f"({file_size_kb:.1f}KB, {prompt_tokens + completion_tokens} tokens)")

            except Exception as e:
                print(f"[ERROR: {str(e)}]")
                continue

    print(f"\n  [✓] Analyzed {len(github_summary)} file(s) total")

    return github_summary


def run():
    """
    Run the profile creation submodule.

    Process:
    1. Read seed profile.md
    2. Extract GitHub profile URL
    3. Analyze pinned repos -> github_summary.json
    4. Create enriched .dxtr/dxtr_profile.md

    Returns:
        str: Path to enriched profile, or None if creation failed
    """
    profile_path = Path("profile.md")

    # Check if seed profile exists
    if not profile_path.exists():
        print("\n" + "=" * 80)
        print("PROFILE NOT FOUND")
        print("=" * 80)
        print("\nNo profile.md found in the current directory.")
        print("\nTo create your profile, please:")
        print("1. Create a file named 'profile.md' in the current directory")
        print("2. Add information about yourself, your experience, and goals")
        print("3. Include your GitHub profile URL (e.g., https://github.com/username)")
        print("4. Restart DXTR to continue\n")
        return None

    print("\n" + "=" * 80)
    print("PROFILE INITIALIZATION")
    print("=" * 80 + "\n")

    # Read seed profile
    profile_content = profile_path.read_text()
    print(f"[Reading seed profile: {len(profile_content)} characters]")

    # Create .dxtr directory
    dxtr_dir = Path(".dxtr")
    dxtr_dir.mkdir(exist_ok=True)

    # Extract and analyze GitHub profile
    github_url = _extract_github_profile_url(profile_content)
    github_summary = {}

    if github_url:
        print(f"[Found GitHub profile URL: {github_url}]")
        github_summary = _analyze_github_repos(github_url)

        # Save github_summary.json
        if github_summary:
            summary_path = dxtr_dir / "github_summary.json"
            summary_path.write_text(json.dumps(github_summary, indent=2))
            print(f"\n[✓] Saved github_summary.json ({len(github_summary)} files)")
        else:
            print("\n[No GitHub analysis data to save]")
    else:
        print("[No GitHub profile URL found in profile.md]")

    # Generate enriched profile
    print("\n" + "=" * 80)
    print("GENERATING ENRICHED PROFILE")
    print("=" * 80 + "\n")

    # Build context for enrichment
    enrichment_context = f"""Original profile:
{profile_content}
"""

    if github_summary:
        # Add summary of GitHub analysis
        num_files = len(github_summary)
        repos = set()
        for file_path in github_summary.keys():
            # Extract repo name from path like .dxtr/repos/owner/repo/...
            parts = Path(file_path).parts
            if 'repos' in parts:
                idx = parts.index('repos')
                if idx + 2 < len(parts):
                    repos.add(f"{parts[idx+1]}/{parts[idx+2]}")

        enrichment_context += f"""

GitHub Analysis:
- Analyzed {num_files} Python files across {len(repos)} repositories
- Repositories: {', '.join(sorted(repos))}
- Detailed analysis saved in .dxtr/github_summary.json
"""

    enrichment_context += """

Please create an enriched profile that:
1. Preserves all information from the original profile
2. Incorporates insights from the GitHub analysis (if available)
3. Maintains a professional, concise format
4. Focuses on demonstrable skills and experience
"""

    print("[Calling LLM to enrich profile...]")

    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a technical profile enrichment assistant. Create clear, professional profiles that highlight demonstrable skills and experience."
            },
            {
                "role": "user",
                "content": enrichment_context
            }
        ],
        options={
            "temperature": 0.3,
            "num_ctx": 16384,
        }
    )

    enriched_profile = response.message.content.strip()

    # Save enriched profile
    output_path = dxtr_dir / "dxtr_profile.md"
    output_path.write_text(enriched_profile)

    print(f"\n[✓] Enriched profile saved to .dxtr/dxtr_profile.md")
    print(f"[✓] Profile initialization complete\n")

    # Echo the profile back to the user
    print("=" * 80)
    print("DXTR PROFILE")
    print("=" * 80 + "\n")
    print(enriched_profile)
    print("\n" + "=" * 80 + "\n")

    return str(output_path)
