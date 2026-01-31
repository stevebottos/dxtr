import os
from zoneinfo import ZoneInfo

blob_store_root = os.getenv("BLOB_STORE_URI", "")

if not blob_store_root:
    raise ValueError("BLOB_STORE_URI not set, dxtr can't do anything without it.")

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
