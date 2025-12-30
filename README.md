<p align="center">
  <img src="assets/logo.png" alt="DXTR Logo" width="600"/>
</p>

# DXTR

**Status: Work in Progress**

DXTR is an AI research assistant for machine learning engineers. It helps you stay current with ML/AI research by intelligently filtering, ranking, and analyzing papers based on your interests and background.

## Features

- **Daily Paper Pipeline**: Automated ETL from HuggingFace daily papers with Docling for PDF processing
- **Personalized Ranking**: Papers ranked based on your profile and GitHub activity
- **Agentic Deep Research**: Multi-step RAG system that generates exploration questions to retrieve relevant paper sections
- **Streaming Agent Architecture**: See agent outputs in real-time as they think and respond
- **GitHub Analysis**: Automatically analyzes your repositories to understand your interests
- **Profile Management**: Maintains personalized context for tailored recommendations

## Installation

### Prerequisites

1. **Install Ollama**: https://ollama.ai
2. **Install Docker**: Required for the Docling-based PDF processing pipeline
3. **Pull required models**:
   ```bash
   ollama pull mistral-nemo      # Main chat agent
   ollama pull gemma3:12b         # Deep research agent
   ollama pull qwen2.5-coder      # Profile creator agent
   ollama pull nomic-embed-text   # Embeddings for RAG
   ```

### Install DXTR

```bash
pip install -e .
```

## Usage

### First-Time Setup

```bash
# Create your profile (analyzes your GitHub repos and interests)
dxtr create-profile

# Process today's papers (downloads, converts PDFs, builds indices)
python process_daily_papers.py
```

### Daily Workflow

```bash
# Start interactive chat
dxtr chat

# In chat, you can:
# - "rank today's papers" - Get personalized paper rankings
# - "what's the best paper today" - See single most relevant paper
# - "do a deep dive on paper 2512.12345" - Agentic RAG analysis
# - "summarize paper X" - Quick summary of a paper
```

### Manual Paper Processing

```bash
# Download and process papers for a specific date
dxtr get-papers --date 2024-12-30

# Update your profile
dxtr create-profile
```

## Architecture

### Multi-Agent System

DXTR uses a streaming multi-agent architecture where each agent outputs directly to the user:

- **Main Agent** (mistral-nemo): Orchestrates tasks and delegates to specialized agents
- **Papers Helper** (mistral-nemo): Ranks papers based on user profile, responds flexibly to queries
- **Deep Research** (gemma3:12b): Agentic RAG system for paper analysis
  - Generates exploration questions from abstract + user context
  - Retrieves relevant chunks via multi-faceted query strategy
  - Synthesizes comprehensive answers using retrieved content
- **Profile Creator** (qwen2.5-coder): Builds and maintains user profiles from GitHub activity
- **Git Helper**: Analyzes repositories to understand user interests

### Key Design Principles

- **Prompts in Markdown**: All prompts live in `prompts/*.md` files, not in code
- **Streaming by Default**: Agent outputs stream in real-time for transparency
- **No Re-synthesis**: Main agent stays silent after tool execution - agent outputs are final
- **User Query Passthrough**: Original user queries passed verbatim to agents

### Papers ETL Pipeline

The `process_daily_papers.py` script orchestrates:
1. Download papers from HuggingFace daily papers
2. Start Docling Docker container for PDF processing
3. Convert PDFs to markdown + layout JSON
4. Build LlamaIndex vector indices with nomic-embed-text
5. Store everything in `.dxtr/hf_papers/YYYY-MM-DD/`

All configuration is centralized in `dxtr/config.py`.

## Development

Run tests:
```bash
pytest
```

## License

MIT
