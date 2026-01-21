import os

blob_store_root = os.getenv("BLOB_STORE_URI", "")

if not blob_store_root:
    raise ValueError("BLOB_STORE_URI not set, dxtr can't do anything without it.")

papers_dir = "daily_papers/{date}"
profiles_dir = "profiles/{user_id}"
