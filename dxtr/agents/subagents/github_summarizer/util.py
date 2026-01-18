import re
import urllib
from pathlib import Path
import subprocess
import shutil


def find_python_files(repo_path: Path, max_files: int = 100) -> list[Path]:
    """Find all Python files in a repository."""
    python_files = []

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
        if any(py_file.match(pattern) for pattern in exclude_patterns):
            continue

        python_files.append(py_file)

        if len(python_files) >= max_files:
            break

    return sorted(python_files)


def extract_pinned_repos(html_content: str) -> list[str]:
    """Extract pinned repository URLs from a GitHub profile page HTML."""
    pattern = r'data-hydro-click="[^"]*PINNED_REPO[^"]*"[^>]*href="(/[^/"]+/[^/"]+)"'
    pattern_reverse = (
        r'href="(/[^/"]+/[^/"]+)"[^>]*data-hydro-click="[^"]*PINNED_REPO[^"]*"'
    )

    repos = []
    seen = set()

    for patt in [pattern, pattern_reverse]:
        matches = re.findall(patt, html_content)
        for match in matches:
            if match not in seen and match.count("/") == 2:
                full_url = f"https://github.com{match}"
                repos.append(full_url)
                seen.add(match)

    return repos


def extract_github_url(profile_content: str) -> str | None:
    """Extract GitHub profile URL from profile.md content."""
    url_pattern = r'https?://github\.com/[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, profile_content)
    for url in urls:
        if is_profile_url(url):
            return url
    return None


def _parse_repo_url(url: str) -> tuple[str, str] | None:
    """Parse a GitHub repository URL to extract owner and repo name."""
    patterns = [
        r"github\.com/([^/]+)/([^/\.]+)",
        r"github\.com/([^/]+)/([^/]+)\.git",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            owner, repo = match.groups()
            if repo.endswith(".git"):
                repo = repo[:-4]
            return owner, repo

    return None


def _get_repo_path(owner: str, repo: str, base_dir: Path) -> Path:
    """Generate the local path for a cloned repository."""
    return base_dir / "repos" / owner / repo


def clone_repo(url: str, base_dir: Path) -> dict:
    """
    Clone a GitHub repository.

    Uses shallow clone (--depth 1) and removes .git directory after cloning.
    Uses caching - if the repo is already cloned, returns success without re-cloning.
    """
    parsed = _parse_repo_url(url)
    if not parsed:
        return {
            "success": False,
            "path": None,
            "message": f"Invalid GitHub repository URL: {url}",
            "url": url,
        }

    owner, repo = parsed
    repo_path = _get_repo_path(owner, repo, base_dir)

    # Check if already cloned
    if repo_path.exists() and repo_path.is_dir():
        return {
            "success": True,
            "path": str(repo_path),
            "message": "Repository already cloned (cached)",
            "url": url,
            "owner": owner,
            "repo": repo,
        }

    # Create parent directory
    repo_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"  [Cloning {owner}/{repo}...]")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, str(repo_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            # Remove .git directory to save space
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
            }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "path": None,
            "message": "Clone timeout (exceeded 120s)",
            "url": url,
        }
    except Exception as e:
        return {
            "success": False,
            "path": None,
            "message": f"Clone error: {str(e)}",
            "url": url,
        }


def fetch_profile_html(url: str) -> str | None:
    """Fetch raw HTML from a GitHub profile URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (DXTR Profile Agent)"}
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, timeout=10) as response:
            content_bytes = response.read()
            content_type = response.headers.get("Content-Type", "")

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


def is_profile_url(url: str) -> bool:
    """Check if a GitHub URL is a profile (not a repository)."""
    pattern = r"github\.com/([^/]+)/?$"
    return bool(re.search(pattern, url))
