"""Paper download and loading utilities for the master agent."""

from tempfile import TemporaryDirectory
from datetime import datetime, timedelta
from pathlib import Path
import json
import asyncio

import requests
from dxtr import constants, util


async def get_available_dates(days_back: int = 7) -> dict[str, int]:
    """Return {date: paper_count} for last N days that have downloaded papers."""
    available = {}

    for i in range(days_back):
        date = (datetime.now(constants.PST) - timedelta(days=i)).strftime("%Y-%m-%d")
        date_dir_str = constants.papers_dir.format(date=date)

        # Check GCS
        try:
            # We list the directory. If it returns items, it exists.
            items = await util.listdir_gcs(date_dir_str)
            # Count subdirectories that likely contain papers (or just count items if structure is flat?)
            # download_papers uploads as: papers_dir/paper_id/metadata.json
            # listdir_gcs returns 'paper_id/' for directories.

            paper_count = 0
            for item in items:
                # If item is a directory (ends with /), check if it has metadata?
                # listdir_gcs returns names relative to prefix.
                # If we have 'paper_id/', we assume it's a paper.
                if item.endswith("/"):
                    paper_count += 1

            if paper_count > 0:
                available[date] = paper_count
        except Exception as e:
            print(f"Error checking date {date}: {e}")

    return available


async def fetch_papers_for_date(date: str) -> list[dict]:
    """Fetch paper metadata from HuggingFace for a given date.

    Returns list of paper metadata dicts with id, title, summary, etc.
    """

    def _fetch():
        try:
            response = requests.get(
                f"{constants.HF_DAILY_PAPERS_URL}?date={date}", timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data
        except Exception as e:
            print(f"HF API Error: {e}")
            return []

    data = await asyncio.to_thread(_fetch)

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
            papers.append(
                {
                    "id": paper_id,
                    "title": paper_data.get("title", ""),
                    "summary": paper_data.get("summary", ""),
                    "authors": paper_data.get("authors", []),
                    "publishedAt": paper_data.get("publishedAt", ""),
                    "upvotes": paper_data.get("upvotes", 0),
                }
            )

    return papers


async def download_papers(
    date: str,
    paper_ids: list[str] | None = None,
    download_pdfs: bool = True,
) -> list[Path]:
    """Download papers from HuggingFace/ArXiv for a date.

    Args:
        date: Date string in YYYY-MM-DD format
        paper_ids: Optional list of specific paper IDs to download. If None, downloads all.
        download_pdfs: Whether to download PDFs (default False per CLAUDE.md - Gemini handles directly)

    Returns:
        List of paths to paper directories
    """
    upload_bucket_root = Path(constants.papers_dir.format(date=date))

    print(f"Fetching papers for {date}...")
    papers = await fetch_papers_for_date(date)
    print(f"{len(papers)} papers found for {date}")
    if not papers:
        return []

    # Filter to specific IDs if provided
    if paper_ids:
        papers = [p for p in papers if p["id"] in paper_ids]

    downloaded = []

    # Process in batches with rate limiting
    with TemporaryDirectory() as tmp:
        tmp_out_dir_root = Path(tmp)

        for i, paper in enumerate(papers):
            paper_id = paper["id"]
            tmp_out_dir = tmp_out_dir_root / paper_id
            tmp_out_dir.mkdir(parents=True)

            # Save metadata
            metadata_path = tmp_out_dir / "metadata.json"
            metadata_path.write_text(json.dumps(paper, indent=2, default=str))
            downloaded.append(metadata_path)

            # Download PDF if requested
            if download_pdfs:
                pdf_path = tmp_out_dir / "paper.pdf"
                if not pdf_path.exists():
                    try:

                        def _download_pdf():
                            r = requests.get(
                                constants.ARXIV_PDF_URL.format(id=paper_id), timeout=60
                            )
                            if r.status_code == 200:
                                pdf_path.write_bytes(r.content)
                                return True
                            return False

                        success = await asyncio.to_thread(_download_pdf)
                        if success:
                            print(f"Downloaded PDF: {paper_id}")
                            downloaded.append(pdf_path)
                            await asyncio.sleep(1)  # Rate limit
                        else:
                            print(f"PDF download failed {paper_id}")
                    except Exception as e:
                        print(f"PDF error {paper_id}: {e}")

            # Rate limit batch processing
            if (i + 1) % 10 == 0:
                await asyncio.sleep(2)

        for f in downloaded:
            # f is like /tmp/.../paper_id/metadata.json
            # We want to upload to papers_dir/date/paper_id/metadata.json

            # f.parts[-2:] gives ('paper_id', 'metadata.json')
            upload_prefix = "/".join(f.parts[-2:])
            full_dest = str(upload_bucket_root / upload_prefix)
            await util.upload_to_gcs(str(f), full_dest)

    return downloaded


async def load_papers_metadata(date: str) -> list[dict]:
    """Load all metadata.json files for a date.

    Returns list of paper metadata dicts.
    """
    date_dir_str = constants.papers_dir.format(date=date)
    data = await util.listdir_gcs(date_dir_str)

    papers = []
    for paper_dir in data:
        # paper_dir is like "paper_id/"
        if not paper_dir.endswith("/"):
            continue

        meta_path = f"{date_dir_str}/{paper_dir}metadata.json"
        # remove double slashes if any
        meta_path = meta_path.replace("//", "/")

        print(f"Loading {meta_path}")
        content = await util.read_from_gcs(meta_path)
        if content:
            metadata = json.loads(content)
            papers.append(metadata)

    return papers


def format_available_dates(available: dict[str, int]) -> str:
    """Format available dates dict into a readable string."""
    if not available:
        return (
            "No papers downloaded yet. Use download_papers to fetch papers for a date."
        )

    lines = ["Available papers:"]
    for date, count in sorted(available.items(), reverse=True):
        lines.append(f"  {date}: {count} papers")

    return "\n".join(lines)


def papers_list_to_dict(papers: list[dict]) -> dict[str, dict]:
    """Convert list of papers to dict keyed by ID."""
    return {
        p["id"]: {"title": p.get("title", ""), "summary": p.get("summary", "")}
        for p in papers
    }


def format_ranking_results(results: list[dict]) -> str:
    """Format ranking results for display.

    Args:
        results: List of {id, title, score, reason} dicts, sorted by score

    Returns:
        Formatted markdown string
    """
    if not results:
        return "No papers ranked."

    lines = ["# Paper Rankings", ""]

    current_tier = None
    for r in results:
        score = r["score"]

        # Determine tier
        if score >= 9:
            tier = "Must Read (9-10)"
        elif score >= 7:
            tier = "Highly Relevant (7-8)"
        elif score >= 5:
            tier = "Moderately Relevant (5-6)"
        elif score >= 3:
            tier = "Low Relevance (3-4)"
        else:
            tier = "Not Relevant (1-2)"

        if tier != current_tier:
            current_tier = tier
            lines.append(f"## {tier}")
            lines.append("")

        lines.append(f"**[{score}/10]** {r['title']}")
        lines.append(f"  - {r['reason']}")
        paper_id = r["id"]
        hf_link = f"https://huggingface.co/papers/{paper_id}"
        arxiv_link = f"https://arxiv.org/abs/{paper_id}"
        lines.append(f"  - [HuggingFace]({hf_link}) | [arXiv]({arxiv_link})")
        lines.append("")

    return "\n".join(lines)
