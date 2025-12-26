<p align="center">
  <img src="assets/logo.png" alt="DXTR Logo" width="600"/>
</p>

# DXTR

**Status: Work in Progress**

DXTR is an AI research assistant for machine learning engineers. It helps you stay current with ML/AI research.

## Features

- Daily paper downloads from HuggingFace's daily papers
- Personalized paper ranking based on your profile and GitHub activity
- Multi-agent architecture with specialized agents for different tasks
- GitHub repository analysis to understand your interests
- Profile creation and maintenance

## Installation

### Prerequisites

1. **Install Ollama**: https://ollama.ai
2. **Pull required models**:
   ```bash
   ollama pull mistral-nemo
   ollama pull gemma3:12b
   ollama pull qwen2.5-coder
   ```

### Install DXTR

```bash
pip install -e .
```

## Usage

```bash
# Start interactive chat
dxtr chat

# Download today's papers
dxtr download-papers

# Create/update your profile
dxtr create-profile
```

## Architecture

DXTR uses a multi-agent system with specialized agents:
- **Main Agent**: Orchestrates tasks and handles general chat
- **Papers Helper**: Ranks and analyzes research papers
- **Profile Creator**: Builds user profiles from GitHub and manual input
- **Git Helper**: Analyzes GitHub repositories for insights

All configuration is centralized in `dxtr/config.py`.

## Development

Run tests:
```bash
pytest
```

## License

MIT
