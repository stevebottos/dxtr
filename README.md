# DXTR

AI research assistant for ML engineers. This project is a work in progress and is live at [https://dxtrchat.app/](https://dxtrchat.app/).

## TODOs

### Completed
- [x] **Multi-Agent Orchestration**: Master agent that intelligently delegates to specialized subagents using `pydantic-ai`.
- [x] **Unified Parallel Executor**: Custom `parallel_map` utility for high-concurrency subagent tasks (e.g., analyzing dozens of files or papers).
- [x] **Real-time UX Feedback**: Event bus architecture that streams tool calls, status updates, and progress via SSE to the frontend.
- [x] **GitHub Analysis**: Capability to clone, parse, and summarize Python repositories to understand a user's technical background.
- [x] **Profile Synthesis**: Automated generation of enriched user profiles from GitHub data and conversation history.
- [x] **Cloud Storage Integration**: Artifact persistence (summaries, profiles, metadata) using Google Cloud Storage.
- [x] **LiteLLM Abstraction**: Unified interface for multiple LLM providers with built-in cost tracking and proxying.
- [x] **Tool Usage Logging**: Decorator-based logging for all agent tool calls to monitor system behavior and model decisions.
- [x] **Infrastructure as Code**: Dockerized setup for consistent development and production environments.
- [x] **Event-Driven Architecture**: Internal event bus using `ContextVar` to track request state and stream updates.
- [x] **Observability**: Built-in tool usage tracking and tracing integration with LiteLLM for performance monitoring.

### Remaining
- [ ] **Paper Ranking Integration**: Fully hook up the Arxiv/HuggingFace paper ranking subagent to the master agent's toolset.
- [ ] **Streaming Structured Outputs**: Enable real-time streaming for Pydantic-based structured outputs (currently SSE only supports status events and final answers).
- [ ] **Long-term Semantic Memory**: Integrate `mem0` or similar for cross-session fact extraction and personalization.
- [ ] **Latency Optimization**: Reduce "time to first token" for conversational turns by optimizing agent initialization and tool selection.
- [ ] **Model Acknowledgment**: Refine system prompts or architecture to ensure the model provides verbal confirmation before executing long-running tools.
- [ ] **Deep Dive Agent**: Specialized agent for in-depth analysis of papers using visual RAG (Retrieval-Augmented Generation).
- [ ] **Evaluation Framework**: Implementation of "Evals" to measure agent performance on retrieval and synthesis tasks.
- [ ] **Code Quality & Testing**: Expand unit and integration test coverage; integrate linting and type-checking (Ruff, Mypy) into CI.

## Requirements

- Docker
- OpenRouter or OpenAI API key
- Google Cloud Storage bucket (for artifact storage)

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-repo/dxtr.git
cd dxtr

# 2. Configure environment variables
cp .env.example .env
# Edit .env with your keys and BLOB_STORE_URI
```

## Development

The project uses a `Makefile` to simplify common operations.

```bash
# Terminal 1: Start infrastructure (LiteLLM)
make up-dev

# Terminal 2: Run backend from source (requires python 3.12+)
make server
```

- **Backend API**: http://localhost:8000
- **LiteLLM Proxy**: http://localhost:4000
- **Health Check**: `curl http://localhost:8000/health`

## Docker Deployment

To run the full stack (including the core API) in Docker:

```bash
# Start everything
make up

# Stop everything
make down
```