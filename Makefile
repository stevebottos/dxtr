.PHONY: up up-dev down server frontend mock-conversation clear-queue build logs test-integration

# =============================================================================
# PRODUCTION (all services in Docker)
# =============================================================================

# Start all services in Docker
up:
	docker compose --profile full up --build

# Stop all services
down:
	docker compose --profile full down

# Build all Docker images
build:
	docker compose --profile full build

# View logs from all services
logs:
	docker compose --profile full logs -f

# =============================================================================
# DEVELOPMENT (infrastructure in Docker, code from source)
# =============================================================================

# Start infrastructure only (db + litellm)
up-dev:
	docker compose up

# Start dxtr server from source (port 8000)
server:
	cd dxtr && python server.py

# Start frontend dev server (port 3000)
frontend:
	cd frontend && npm install && npm run dev

# Run mock conversation (requires server running)
mock-conversation:
	python mock_conversation.py

deploy-dxtr:
	docker tag dxtr-core:latest gcr.io/$$(GCLOUD_PROJECT_ID)/dxtr-core:latest
	docker push gcr.io/$$(GCLOUD_PROJECT_ID)/dxtr-core:latest
# =============================================================================
# UTILITIES
# =============================================================================

# Clear LiteLLM request queue (run while docker is up)
clear-queue:
	@echo "Clearing LiteLLM request queue..."
	@docker exec litellm_db psql -U llm_admin -d litellm_db -c "TRUNCATE litellm_proxy_request_queue CASCADE;" 2>/dev/null || true
	@docker exec litellm_db psql -U llm_admin -d litellm_db -c "DELETE FROM litellm_spendlogs WHERE status = 'pending';" 2>/dev/null || true
	@echo "Queue cleared."

# Pull latest LiteLLM image
pull-litellm:
	docker pull docker.litellm.ai/berriai/litellm:main-latest

# =============================================================================
# TESTING
# =============================================================================

# Run integration tests (requires services running)
test-integration:
	pytest -m integration -v
