"""
Centralized configuration for DXTR.

All configuration values in one place:
- Model selections per agent
- LLM parameters (temperature, context window)
- Directory paths
- Tool settings
"""

from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for a specific agent's model."""

    name: str
    temperature: float = 0.3
    context_window: int = 16384
    options: dict = field(default_factory=dict)


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
class Config:
    """Main DXTR configuration."""

    # Model configurations per agent
    models: dict[str, ModelConfig] = field(
        default_factory=lambda: {
            "main": ModelConfig(
                name="mistral-nemo",
                temperature=0.3,
                context_window=16384,
            ),
            "profile_creator": ModelConfig(
                name="gemma3:12b",
                temperature=0.3,
                context_window=16384,
            ),
            "git_helper": ModelConfig(
                name="qwen2.5-coder",
                temperature=0.3,
                context_window=16384,
            ),
            "papers_helper": ModelConfig(
                name="gemma3:12b",
                temperature=0.1,
                context_window=16384,
            ),
            "deep_research": ModelConfig(
                name="nemotron-3-nano",
                temperature=0.3,
                context_window=131072,  # 128k - plenty for full papers (model supports 1M)
            ),
        }
    )

    # Paths
    paths: PathConfig = field(default_factory=PathConfig)

    # Git helper settings
    git_cache_ttl_days: int = 7
    git_max_files_per_repo: int = 50

    # Papers settings
    papers_max_results: int = 50

    def get_model_config(self, agent_name: str) -> ModelConfig:
        """
        Get model configuration for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            ModelConfig: The model configuration

        Raises:
            KeyError: If agent not found
        """
        if agent_name not in self.models:
            raise KeyError(
                f"No model config for agent '{agent_name}'. "
                f"Available: {list(self.models.keys())}"
            )
        return self.models[agent_name]


# Global config instance
config = Config()
