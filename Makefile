.PHONY: litellm agent-server mock-conversation dev stop clear-queue

# Start LiteLLM proxy (port 4000)
litellm:
	litellm --config dxtr/litellm_config.yaml --port 4000

# Start agent server (port 8000)
agent-server:
	cd dxtr && python server.py

# Run mock conversation (requires both servers running)
mock-conversation:
	python mock_conversation.py

get-litellm:
	docker pull docker.litellm.ai/berriai/litellm:main-latest

start-litellm:
	docker run \
    -v $$(pwd)/litellm_config.yaml:/app/config.yaml \
		-e LITELLM_MASTER_KEY=sk-1234 \
		-e OPENROUTER_API_KEY=$${OPENROUTER_API_KEY} \
    -p 4000:4000 \
    docker.litellm.ai/berriai/litellm:main-latest \
    --config /app/config.yaml --detailed_debug

server:
	@echo "Starting agent server on :8000..."
	@cd dxtr && python server.py || (cd .. && make stop && exit 1)

# Clear LiteLLM request queue (run while docker is up)
clear-queue:
	@echo "Clearing LiteLLM request queue..."
	@docker exec litellm_db psql -U llm_admin -d litellm_db -c "TRUNCATE litellm_proxy_request_queue CASCADE;" 2>/dev/null || true
	@docker exec litellm_db psql -U llm_admin -d litellm_db -c "DELETE FROM litellm_spendlogs WHERE status = 'pending';" 2>/dev/null || true
	@echo "Queue cleared."
