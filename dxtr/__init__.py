from dotenv import load_dotenv

load_dotenv()

from pathlib import Path


def load_system_prompt(file_path: Path) -> str:
    """Load a system prompt from a markdown file."""
    return file_path.read_text().strip()
