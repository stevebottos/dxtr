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
from ..agent import MODEL, _load_system_prompt
from ...git_helper import agent as git_helper
from ...git_helper.tools import git_tools


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

    Delegates to the git_helper agent.

    Args:
        github_url: GitHub profile URL

    Returns:
        Dict mapping file paths to JSON analysis strings
    """
    # Delegate to git_helper agent
    return git_helper.run(github_url)


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
    enrichment_context = f"""# Original Profile

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

# GitHub Analysis Summary

- **Files analyzed**: {num_files} Python files
- **Repositories**: {len(repos)} repositories
  - {chr(10).join([f'  - {repo}' for repo in sorted(repos)])}
- **Detailed analysis**: Available in .dxtr/github_summary.json
"""
    else:
        enrichment_context += "\n# GitHub Analysis Summary\n\nNo GitHub profile found or no repositories analyzed.\n"

    print("[Calling LLM to enrich profile...]")

    # Load system prompt
    system_prompt = _load_system_prompt("profile_creation")

    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
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
