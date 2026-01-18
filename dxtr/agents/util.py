"""Paper download and loading utilities for the master agent."""

from datetime import datetime, timedelta
from pathlib import Path
import json
import time

import requests

from dxtr import DXTR_DIR


PAPERS_DIR = DXTR_DIR / "papers"
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

ARXIV_PDF_URL = "https://arxiv.org/pdf/{id}"
HF_DAILY_PAPERS_URL = "https://huggingface.co/api/daily_papers"


def get_available_dates(days_back: int = 7) -> dict[str, int]:
    """Return {date: paper_count} for last N days that have downloaded papers."""
    available = {}

    for i in range(days_back):
        date = (datetime.today() - timedelta(days=i)).strftime("%Y-%m-%d")
        date_dir = PAPERS_DIR / date

        if date_dir.exists():
            # Count papers (subdirectories with metadata.json)
            paper_count = sum(
                1 for p in date_dir.iterdir()
                if p.is_dir() and (p / "metadata.json").exists()
            )
            if paper_count > 0:
                available[date] = paper_count

    return available


def fetch_papers_for_date(date: str) -> list[dict]:
    """Fetch paper metadata from HuggingFace for a given date.

    Returns list of paper metadata dicts with id, title, summary, etc.
    """
    try:
        response = requests.get(f"{HF_DAILY_PAPERS_URL}?date={date}", timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"HF API Error: {e}")
        return []

    if not isinstance(data, list):
        print("Invalid HF response")
        return []

    # Normalize the data structure
    papers = []
    for item in data:
        # HF API sometimes nests paper data under 'paper' key
        paper_data = item.get("paper", item)
        paper_id = paper_data.get("id")
        if paper_id:
            papers.append({
                "id": paper_id,
                "title": paper_data.get("title", ""),
                "summary": paper_data.get("summary", ""),
                "authors": paper_data.get("authors", []),
                "publishedAt": paper_data.get("publishedAt", ""),
                "upvotes": item.get("upvotes", 0),
            })

    return papers


def download_papers(
    date: str,
    paper_ids: list[str] | None = None,
    download_pdfs: bool = False,
) -> list[Path]:
    """Download papers from HuggingFace/ArXiv for a date.

    Args:
        date: Date string in YYYY-MM-DD format
        paper_ids: Optional list of specific paper IDs to download. If None, downloads all.
        download_pdfs: Whether to download PDFs (default False per CLAUDE.md - Gemini handles directly)

    Returns:
        List of paths to paper directories
    """
    out_dir = PAPERS_DIR / date
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching papers for {date}...")
    papers = fetch_papers_for_date(date)

    if not papers:
        print(f"No papers found for {date}")
        return []

    # Filter to specific IDs if provided
    if paper_ids:
        papers = [p for p in papers if p["id"] in paper_ids]

    downloaded = []

    # Process in batches with rate limiting
    for i, paper in enumerate(papers):
        paper_id = paper["id"]
        paper_dir = out_dir / paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata_path = paper_dir / "metadata.json"
        metadata_path.write_text(json.dumps(paper, indent=2, default=str))

        # Download PDF if requested
        if download_pdfs:
            pdf_path = paper_dir / "paper.pdf"
            if not pdf_path.exists():
                try:
                    r = requests.get(ARXIV_PDF_URL.format(id=paper_id), timeout=60)
                    if r.status_code == 200:
                        pdf_path.write_bytes(r.content)
                        print(f"Downloaded PDF: {paper_id}")
                        time.sleep(1)  # Rate limit
                    else:
                        print(f"PDF download failed {paper_id}: {r.status_code}")
                except Exception as e:
                    print(f"PDF error {paper_id}: {e}")

        downloaded.append(paper_dir)

        # Rate limit batch processing
        if (i + 1) % 10 == 0:
            time.sleep(1)

    print(f"Downloaded {len(downloaded)} papers for {date}")
    return downloaded


def load_papers_metadata(date: str) -> list[dict]:
    """Load all metadata.json files for a date.

    Returns list of paper metadata dicts.
    """
    date_dir = PAPERS_DIR / date

    if not date_dir.exists():
        return []

    papers = []
    for paper_dir in date_dir.iterdir():
        if not paper_dir.is_dir():
            continue

        metadata_path = paper_dir / "metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
                papers.append(metadata)
            except json.JSONDecodeError:
                print(f"Invalid metadata.json in {paper_dir}")
                continue

    return papers


def format_available_dates(available: dict[str, int]) -> str:
    """Format available dates dict into a readable string."""
    if not available:
        return "No papers downloaded yet. Use download_papers to fetch papers for a date."

    lines = ["Available papers:"]
    for date, count in sorted(available.items(), reverse=True):
        lines.append(f"  {date}: {count} papers")

    return "\n".join(lines)
