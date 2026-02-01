"""Application constants."""

import os

# Server config
KEEPALIVE_INTERVAL_SECONDS = 10

# LiteLLM
BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")
API_KEY = os.environ.get("LITELLM_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "LITELLM_API_KEY not set. See .env.example for required configuration."
    )
