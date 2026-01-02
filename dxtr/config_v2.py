"""
DXTR Configuration V2 - SGLang Edition

Simplified configuration for single-model SGLang architecture.
"""

from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class SGLangConfig:
    """SGLang server configuration."""

    base_url: str = "http://localhost:30000/v1"
    model_name: str = "default"  # SGLang uses "default" identifier
    temperature: float = 0.3
    max_tokens: int = 2000
    context_window: int = 32768  # 32K context


@dataclass
class EmbeddingConfig:
    """Ollama embedding configuration (unchanged)."""

    model_name: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"


@dataclass
class PathConfig:
    """Paths used throughout DXTR."""

    dxtr_dir: Path = Path(".dxtr")
    repos_dir: Path = Path(".dxtr/repos")
    papers_dir: Path = Path(".dxtr/hf_papers")
    profile_file: Path = Path(".dxtr/dxtr_profile.md")
    github_summary_file: Path = Path(".dxtr/github_summary.json")
    seed_profile_file: Path = Path("profile.md")


@dataclass
class AgentConfig:
    """Agent-specific configuration overrides."""

    # Profile manager - needs structured output
    profile_manager_max_tokens: int = 2000
    profile_manager_temperature: float = 0.3

    # Deep research - needs longer responses
    deep_research_max_tokens: int = 1000
    deep_research_temperature: float = 0.3

    # Papers helper - deterministic ranking
    papers_helper_temperature: float = 0.0
    papers_helper_max_tokens: int = 1500


@dataclass
class Config:
    """Main DXTR configuration."""

    sglang: SGLangConfig = field(default_factory=SGLangConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)

    # General settings
    git_cache_ttl_days: int = 7
    git_max_files_per_repo: int = 50
    papers_max_results: int = 50


# Global config instance
config = Config()
