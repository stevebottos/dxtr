# DXTR

AI research assistant for ML engineers. Work in progress.

## Requirements

- Docker
- OpenRouter API key

## Setup

```bash
cp .env.example .env
# Edit .env with your keys
```

## Development

```bash
# Terminal 1: Start infrastructure (postgres + litellm)
make up-dev

# Terminal 2: Run backend from source
make server

# Terminal 3: Run frontend from source
make frontend
```

- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- LiteLLM: http://localhost:4000

## Full Docker

```bash
# Start everything
make up

# Stop everything
make down
```
