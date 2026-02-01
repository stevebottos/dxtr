import os
from zoneinfo import ZoneInfo

papers_dir = "daily_papers/{date}"
profiles_dir = "profiles/{user_id}"

# Timezone for date operations (HuggingFace daily papers use PST)
PST = ZoneInfo("America/Los_Angeles")

# External API URLs
ARXIV_PDF_URL = "https://arxiv.org/pdf/{id}"
HF_DAILY_PAPERS_URL = "https://huggingface.co/api/daily_papers"

# Server config
KEEPALIVE_INTERVAL_SECONDS = 10

# Paper ranking
MAX_PAPERS_TO_RANK = 60

BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")
API_KEY = os.environ.get("LITELLM_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "LITELLM_API_KEY not set. See .env.example for required configuration."
    )
