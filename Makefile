.PHONY: up down server frontend clear-queue build logs test-integration papers-service papers-service-stop push deploy-dxtr litellm pull-litellm

# =============================================================================
# PRODUCTION (all services in Docker)
# =============================================================================

# Start all services in Docker
up:
	docker compose up --build

# Stop all services
down:
	docker compose down

# Build all Docker images
build:
	docker compose build

push: build
	docker compose push

server:
	cd dxtr && python server.py

litellm:
	docker compose up litellm
# Start frontend dev server (port 3000)
frontend:
	cd frontend && npm install && npm run dev

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


# =============================================================================
# PAPERS SERVICE
# =============================================================================

# Run papers update service (hourly cron, 3-day lookback)
papers-service:
	docker build -f docker/Dockerfile.papers-service -t papers-service .
	docker run -d --name papers-service \
		-e DATABASE_URL=$$(grep DATABASE_URL .env | cut -d= -f2- | tr -d '"') \
		papers-service
	@echo "Papers service running in background. Logs: docker logs -f papers-service"

papers-service-stop:
	docker stop papers-service && docker rm papers-service
