"""
Repository Analyzer Submodule

Analyzes cloned repositories using LLM-based module-level analysis.
"""

import hashlib
import json
from pathlib import Path
from typing import Any
from ollama import chat

# Model for fast code analysis
ANALYSIS_MODEL = "qwen2.5-coder"

# JSON schema for structured module analysis
MODULE_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "keywords": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["keywords", "summary"],
}


def _get_analysis_cache_path(repo_path: str) -> Path:
    """
    Generate cache path for repository analysis.

    Args:
        repo_path: Path to the repository

    Returns:
        Path to cache file
    """
    # Create hash from repo path
    path_hash = hashlib.md5(repo_path.encode()).hexdigest()[:8]
    # Extract repo name from path (e.g., .dxtr/repos/owner/repo -> owner_repo)
    parts = Path(repo_path).parts
    if len(parts) >= 2:
        cache_name = f"analysis_{parts[-2]}_{parts[-1]}_{path_hash}.md"
    else:
        cache_name = f"analysis_{path_hash}.md"

    return Path(".dxtr") / cache_name




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


def _analyze_module(file_path: Path) -> dict[str, Any]:
    """
    Generate LLM analysis of a Python module.

    Args:
        file_path: Path to Python file

    Returns:
        Dict with 'keywords' list and 'summary' string
    """
    try:
        source_code = file_path.read_text(encoding='utf-8')
    except Exception as e:
        return {
            'keywords': [],
            'summary': f"Error reading file: {str(e)}"
        }

    # System prompt for module analysis
    system_prompt = """Assess this Python module for relevant experience, technologies, techniques, and implementations.

Focus on:
- Specific libraries/frameworks used (e.g., "torch.nn", "transformers", "scann")
- Modern techniques present or absent (e.g., "ROPE", "flash attention", "KV caching")
- Implementation patterns (e.g., "custom attention", "CUDA kernels", "NumPy-based")
- Architectural approaches (e.g., "encoder-decoder", "transformer", "CNN")

Output:
1. Comprehensive list of keywords (technical terms, libraries, techniques)
2. Brief module-level summary (2-3 sentences)"""

    try:
        response = chat(
            model=ANALYSIS_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": source_code}
            ],
            options={
                "temperature": 0.3,
                "num_ctx": 16384,
            },
            format=MODULE_ANALYSIS_SCHEMA
        )

        # Parse JSON response
        result = json.loads(response.message.content)
        return {
            'keywords': result.get('keywords', []),
            'summary': result.get('summary', '')
        }

    except Exception as e:
        return {
            'keywords': [],
            'summary': f"Analysis error: {str(e)}"
        }


def analyze_repository(repo_path: str | Path, summarize: bool = True) -> dict[str, Any]:
    """
    Analyze an entire repository.

    Args:
        repo_path: Path to repository
        summarize: Whether to generate LLM summaries for functions

    Returns:
        Dictionary with repository analysis
    """
    repo_path = Path(repo_path)

    if not repo_path.exists():
        return {'error': f'Repository path does not exist: {repo_path}'}

    print(f"\n[Analyzing repository: {repo_path.name}]")

    # Find Python files
    python_files = find_python_files(repo_path)
    print(f"  [Found {len(python_files)} Python file(s)]")

    # Analyze each file
    file_analyses = []
    for py_file in python_files:
        rel_path = py_file.relative_to(repo_path)
        print(f"  [Analyzing: {rel_path}]")

        if summarize:
            # LLM-based module analysis
            module_analysis = _analyze_module(py_file)
            file_analyses.append({
                'relative_path': str(rel_path),
                'keywords': module_analysis['keywords'],
                'summary': module_analysis['summary']
            })
        else:
            # Just track the file
            file_analyses.append({
                'relative_path': str(rel_path),
                'keywords': [],
                'summary': ''
            })

    return {
        'repo_path': str(repo_path),
        'repo_name': repo_path.name,
        'total_files': len(python_files),
        'analyzed_files': len(file_analyses),
        'files': file_analyses,
    }


def format_analysis_as_markdown(analysis: dict[str, Any]) -> str:
    """
    Format repository analysis as markdown.

    Args:
        analysis: Analysis dictionary from analyze_repository()

    Returns:
        Markdown-formatted string
    """
    lines = []

    lines.append(f"# Repository Analysis: {analysis['repo_name']}\n")
    lines.append(f"**Path:** `{analysis['repo_path']}`")
    lines.append(f"**Files analyzed:** {analysis['analyzed_files']} / {analysis['total_files']}\n")
    lines.append("---\n")

    for file_data in analysis['files']:
        rel_path = file_data['relative_path']
        lines.append(f"## File: `{rel_path}`\n")

        # Module-level summary
        if file_data.get('summary'):
            lines.append(f"**Summary:** {file_data['summary']}\n")

        # Keywords
        if file_data.get('keywords'):
            keywords_str = ", ".join([f"`{k}`" for k in file_data['keywords'][:15]])
            if len(file_data['keywords']) > 15:
                keywords_str += f" ... and {len(file_data['keywords']) - 15} more"
            lines.append(f"**Keywords:** {keywords_str}\n")

        lines.append("---\n")

    return "\n".join(lines)
