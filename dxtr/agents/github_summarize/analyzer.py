"""
Repository Analyzer Utilities

Utility functions for analyzing Python repositories.
"""

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

# JSON schema for verification of generated summaries
VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "keywords_score": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
        "keywords_issues": {"type": "array", "items": {"type": "string"}},
        "summary_score": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
        "summary_issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["keywords_score", "summary_score"],
}


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
        '*/test/*', '*/tests/*',
        '*/__pycache__/*',
        '*/venv/*', '*/env/*', '*/.venv/*',
        '*/node_modules/*',
        '*/.git/*',
        '*/dist/*', '*/build/*',
        '*/.pytest_cache/*',
    ]

    for py_file in repo_path.rglob('*.py'):
        # Check if file matches any exclude pattern
        if any(py_file.match(pattern) for pattern in exclude_patterns):
            continue

        python_files.append(py_file)

        if len(python_files) >= max_files:
            break

    return sorted(python_files)
