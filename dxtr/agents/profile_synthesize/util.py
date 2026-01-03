"""
Git Tools Module

NOTE: These are utility functions, NOT Ollama function calling tools.
No TOOL_DEFINITION exists for these - they are called directly by the git_helper agent.

Provides functionality to clone and cache GitHub repositories.
"""

import subprocess
import re
import shutil
import urllib.request
from pathlib import Path


# JSON schema for structured module analysis
MODULE_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "keywords": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["keywords", "summary"],
}

# # JSON schema for verification of generated summaries
# VERIFICATION_SCHEMA = {
#     "type": "object",
#     "properties": {
#         "keywords_score": {
#             "type": "integer",
#             "minimum": 1,
#             "maximum": 5,
#         },
#         "keywords_issues": {"type": "array", "items": {"type": "string"}},
#         "summary_score": {
#             "type": "integer",
#             "minimum": 1,
#             "maximum": 5,
#         },
#         "summary_issues": {"type": "array", "items": {"type": "string"}},
#     },
#     "required": ["keywords_score", "summary_score"],
# }


def find_python_files(repo_path: Path, max_files: int = 100) -> list[Path]:
    """
    Find all Python files in a repository.

    Args:
        repo_path: Path to repository
        max_files: Maximum number of files to analyze

    Returns:
        List of Python file paths
    """
    python_files = []

    # Patterns to exclude
    exclude_patterns = [
        "*/test/*",
        "*/tests/*",
        "*/__pycache__/*",
        "*/venv/*",
        "*/env/*",
        "*/.venv/*",
        "*/node_modules/*",
        "*/.git/*",
        "*/dist/*",
        "*/build/*",
        "*/.pytest_cache/*",
    ]

    for py_file in repo_path.rglob("*.py"):
        # Check if file matches any exclude pattern
        if any(py_file.match(pattern) for pattern in exclude_patterns):
            continue

        python_files.append(py_file)

        if len(python_files) >= max_files:
            break

    return sorted(python_files)


def is_profile_url(url: str) -> bool:
    """
    Check if a GitHub URL is a profile (not a repository).

    Args:
        url: GitHub URL

    Returns:
        bool: True if it's a profile URL (e.g., github.com/username)
    """
    # Profile URLs have format: github.com/username (no additional path)
    # Repo URLs have format: github.com/username/repo
    pattern = r"github\.com/([^/]+)/?$"
    return bool(re.search(pattern, url))


def fetch_profile_html(url: str) -> str | None:
    """
    Fetch raw HTML from a GitHub profile URL.

    Args:
        url: The GitHub profile URL to fetch

    Returns:
        Raw HTML string or None if failed
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (DXTR Profile Agent)"}
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, timeout=10) as response:
            content_bytes = response.read()
            content_type = response.headers.get("Content-Type", "")

            # Try to detect encoding
            encoding = "utf-8"
            if "charset=" in content_type:
                encoding = content_type.split("charset=")[-1].split(";")[0].strip()

            try:
                html = content_bytes.decode(encoding)
            except UnicodeDecodeError:
                html = content_bytes.decode("utf-8", errors="ignore")

            return html

    except Exception as e:
        print(f"  [Error fetching profile HTML: {e}]")
        return None


def extract_pinned_repos(html_content: str) -> list[str]:
    """
    Extract pinned repository URLs from a GitHub profile page HTML.

    Args:
        html_content: Raw HTML from GitHub profile page

    Returns:
        List of full repository URLs
    """
    # GitHub pinned repos have data-hydro-click with "target":"PINNED_REPO"
    # Pattern: look for links with PINNED_REPO in data-hydro-click and extract href
    pattern = r'data-hydro-click="[^"]*PINNED_REPO[^"]*"[^>]*href="(/[^/"]+/[^/"]+)"'

    # Also try the reverse order (href before data-hydro-click)
    pattern_reverse = (
        r'href="(/[^/"]+/[^/"]+)"[^>]*data-hydro-click="[^"]*PINNED_REPO[^"]*"'
    )

    repos = []
    seen = set()

    # Try both patterns
    for patt in [pattern, pattern_reverse]:
        matches = re.findall(patt, html_content)
        for match in matches:
            # match is like: /username/repo-name
            if (
                match not in seen and match.count("/") == 2
            ):  # Ensure it's /owner/repo format
                full_url = f"https://github.com{match}"
                repos.append(full_url)
                seen.add(match)

    return repos


def _parse_repo_url(url: str) -> tuple[str, str] | None:
    """
    Parse a GitHub repository URL to extract owner and repo name.

    Args:
        url: GitHub repository URL (e.g., https://github.com/owner/repo)

    Returns:
        Tuple of (owner, repo_name) or None if not a valid repo URL
    """
    # Handle various GitHub URL formats
    patterns = [
        r"github\.com/([^/]+)/([^/\.]+)",  # https://github.com/owner/repo
        r"github\.com/([^/]+)/([^/]+)\.git",  # https://github.com/owner/repo.git
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            owner, repo = match.groups()
            # Remove .git suffix if present (using removesuffix for Python 3.9+)
            if repo.endswith(".git"):
                repo = repo[:-4]
            return owner, repo

    return None


def _get_repo_path(owner: str, repo: str) -> Path:
    """
    Generate the local path for a cloned repository.

    Args:
        owner: Repository owner
        repo: Repository name

    Returns:
        Path object for the repository directory
    """
    return Path(".dxtr") / "repos" / owner / repo


def is_repo_cloned(url: str) -> tuple[bool, Path | None]:
    """
    Check if a repository is already cloned.

    Args:
        url: GitHub repository URL

    Returns:
        Tuple of (is_cloned, repo_path)
    """
    parsed = _parse_repo_url(url)
    if not parsed:
        return False, None

    owner, repo = parsed
    repo_path = _get_repo_path(owner, repo)

    # Check if directory exists (no need to check for .git since we remove it)
    if repo_path.exists() and repo_path.is_dir():
        return True, repo_path

    return False, repo_path


def clone_repo(url: str) -> dict[str, any]:
    """
    Clone a GitHub repository to .dxtr/repos/.

    Uses shallow clone (--depth 1) and removes .git directory after cloning
    to save space. Only source code is kept, no git history.

    Uses caching - if the repo is already cloned, returns success without re-cloning.

    Args:
        url: GitHub repository URL

    Returns:
        Dict with keys:
            - success: bool indicating if clone succeeded
            - path: Path to cloned repo (if successful)
            - message: Status message
            - url: Original URL
            - owner: Repository owner
            - repo: Repository name
    """
    # Check if already cloned
    is_cloned, repo_path = is_repo_cloned(url)

    if is_cloned:
        parsed = _parse_repo_url(url)
        owner, repo = parsed
        return {
            "success": True,
            "path": str(repo_path),
            "message": f"Repository already cloned (cached)",
            "url": url,
            "owner": owner,
            "repo": repo,
        }

    # Parse URL
    parsed = _parse_repo_url(url)
    if not parsed:
        return {
            "success": False,
            "path": None,
            "message": f"Invalid GitHub repository URL: {url}",
            "url": url,
            "owner": None,
            "repo": None,
        }

    owner, repo = parsed
    repo_path = _get_repo_path(owner, repo)

    # Create parent directory
    repo_path.parent.mkdir(parents=True, exist_ok=True)

    # Clone the repository
    try:
        print(f"  [Cloning {owner}/{repo}...]")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, str(repo_path)],
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
        )

        if result.returncode == 0:
            # Remove .git directory to save space (we don't need git history)
            git_dir = repo_path / ".git"
            if git_dir.exists():
                shutil.rmtree(git_dir)

            return {
                "success": True,
                "path": str(repo_path),
                "message": f"Successfully cloned {owner}/{repo}",
                "url": url,
                "owner": owner,
                "repo": repo,
            }
        else:
            return {
                "success": False,
                "path": None,
                "message": f"Git clone failed: {result.stderr}",
                "url": url,
                "owner": owner,
                "repo": repo,
            }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "path": None,
            "message": f"Clone timeout (exceeded 120s)",
            "url": url,
            "owner": owner,
            "repo": repo,
        }
    except Exception as e:
        return {
            "success": False,
            "path": None,
            "message": f"Clone error: {str(e)}",
            "url": url,
            "owner": owner,
            "repo": repo,
        }


def clone_repos(urls: list[str]) -> list[dict[str, any]]:
    """
    Clone multiple repositories.

    Args:
        urls: List of GitHub repository URLs

    Returns:
        List of result dicts from clone_repo()
    """
    results = []
    for url in urls:
        result = clone_repo(url)
        results.append(result)

    return results
