"""Service for fetching and updating paper metadata from HuggingFace."""

import argparse
import asyncio
import json
import logging
import os
from datetime import date, timedelta

import pandas as pd
import psycopg2
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")

HF_DAILY_PAPERS_URL = "https://huggingface.co/api/daily_papers"

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 60


async def fetch_papers_metadata(target_date: date) -> pd.DataFrame:
    """Fetch paper metadata from HuggingFace API for a given date.

    Args:
        target_date: Date to fetch papers for

    Returns:
        DataFrame with columns: id, title, summary, authors, publishedAt, upvotes
        Empty DataFrame if all retries fail (3 attempts, 1 min between).
    """
    date_str = target_date.strftime("%Y-%m-%d")
    columns = ["id", "title", "summary", "authors", "publishedAt", "upvotes"]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data = await _fetch_from_hf(date_str)
            papers = _normalize_response(data)

            if not papers:
                logger.warning(f"No papers found for {date_str}")
                return pd.DataFrame(columns=columns)

            return pd.DataFrame(papers)

        except NoRetryError as e:
            logger.info(f"No papers for {date_str} (client error: {e})")
            return pd.DataFrame(columns=columns)

        except Exception as e:
            logger.error(f"Attempt {attempt}/{MAX_RETRIES} failed for {date_str}: {e}")

            if attempt < MAX_RETRIES:
                logger.info(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                await asyncio.sleep(RETRY_DELAY_SECONDS)

    logger.error(f"All {MAX_RETRIES} attempts failed for {date_str}, returning empty DataFrame")
    return pd.DataFrame(columns=columns)


async def fetch_papers_for_period(lookback_days: int) -> pd.DataFrame:
    """Fetch paper metadata for a lookback period starting from today.

    Args:
        lookback_days: Number of days to look back (including today)

    Returns:
        Combined DataFrame with all papers, with 'date' column added.
        Papers from dates that failed to fetch are excluded.
    """
    today = date.today()
    all_dfs = []

    for days_ago in range(lookback_days):
        target_date = today - timedelta(days=days_ago)
        logger.info(f"Fetching papers for {target_date}")

        df = await fetch_papers_metadata(target_date)

        if not df.empty:
            df["date"] = target_date
            all_dfs.append(df)

    if not all_dfs:
        columns = ["id", "title", "summary", "authors", "publishedAt", "upvotes", "date"]
        return pd.DataFrame(columns=columns)

    return pd.concat(all_dfs, ignore_index=True)


class NoRetryError(Exception):
    """Error that should not be retried (e.g., 4xx client errors)."""
    pass


async def _fetch_from_hf(date_str: str) -> list[dict]:
    """Fetch raw data from HuggingFace API."""

    def _request():
        response = requests.get(f"{HF_DAILY_PAPERS_URL}?date={date_str}", timeout=30)
        if response.status_code >= 400 and response.status_code < 500:
            raise NoRetryError(f"{response.status_code} for {date_str}")
        response.raise_for_status()
        return response.json()

    data = await asyncio.to_thread(_request)

    if not isinstance(data, list):
        raise ValueError(f"Invalid response from HuggingFace API: expected list, got {type(data)}")

    return data


def _normalize_response(data: list[dict]) -> list[dict]:
    """Normalize HuggingFace API response into consistent paper records."""
    papers = []

    for item in data:
        # HF API nests paper data under 'paper' key
        paper_data = item.get("paper", item)
        paper_id = paper_data.get("id")

        if not paper_id:
            continue

        papers.append({
            "id": paper_id,
            "title": paper_data.get("title", ""),
            "summary": paper_data.get("summary", ""),
            "authors": paper_data.get("authors", []),
            "publishedAt": paper_data.get("publishedAt", ""),
            "upvotes": item.get("upvotes", 0),
        })

    return papers


def get_existing_papers(paper_ids: list[str]) -> dict[str, dict]:
    """Fetch existing papers from DB by IDs.

    Returns:
        Dict mapping paper_id -> {upvotes, title, summary, ...}
    """
    if not paper_ids:
        return {}

    conn = psycopg2.connect(DATABASE_URL)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, title, summary, authors, published_at, upvotes, date FROM papers WHERE id = ANY(%s)",
            (paper_ids,),
        )
        rows = cur.fetchall()
        cur.close()

        return {
            row[0]: {
                "id": row[0],
                "title": row[1],
                "summary": row[2],
                "authors": row[3],
                "published_at": row[4],
                "upvotes": row[5],
                "date": row[6],
            }
            for row in rows
        }
    finally:
        conn.close()


def upsert_papers(df: pd.DataFrame) -> tuple[int, int]:
    """Upsert papers into the database.

    Returns:
        Tuple of (inserted_count, updated_count)
    """
    if df.empty:
        return 0, 0

    conn = psycopg2.connect(DATABASE_URL)
    inserted = 0
    updated = 0

    try:
        cur = conn.cursor()

        for _, row in df.iterrows():
            cur.execute(
                """
                INSERT INTO papers (id, title, summary, authors, published_at, upvotes, date)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    summary = EXCLUDED.summary,
                    authors = EXCLUDED.authors,
                    published_at = EXCLUDED.published_at,
                    upvotes = EXCLUDED.upvotes,
                    date = EXCLUDED.date,
                    updated_at = NOW()
                RETURNING (xmax = 0) AS inserted
                """,
                (
                    row["id"],
                    row["title"],
                    row["summary"],
                    json.dumps(row["authors"]),
                    row["publishedAt"] or None,
                    row["upvotes"],
                    row["date"],
                ),
            )
            was_inserted = cur.fetchone()[0]
            if was_inserted:
                inserted += 1
            else:
                updated += 1

        conn.commit()
        cur.close()
        return inserted, updated
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def preview_changes(df: pd.DataFrame) -> None:
    """Preview what changes would be made to the database."""
    if df.empty:
        print("No papers to process.")
        return

    paper_ids = df["id"].tolist()
    existing = get_existing_papers(paper_ids)

    new_papers = []
    updated_papers = []

    for _, row in df.iterrows():
        paper_id = row["id"]
        if paper_id not in existing:
            new_papers.append(row)
        else:
            old = existing[paper_id]
            # Check if any field changed
            if (
                old["upvotes"] != row["upvotes"]
                or old["title"] != row["title"]
                or old["summary"] != row["summary"]
            ):
                updated_papers.append({"old": old, "new": row})

    print(f"\n{'='*60}")
    print(f"PREVIEW (--update-db not set)")
    print(f"{'='*60}")
    print(f"Total papers fetched: {len(df)}")
    print(f"New papers to insert: {len(new_papers)}")
    print(f"Papers to update: {len(updated_papers)}")
    print(f"Unchanged: {len(df) - len(new_papers) - len(updated_papers)}")

    if new_papers:
        print(f"\n--- New Papers (showing first 5) ---")
        for paper in new_papers[:5]:
            print(f"  [{paper['id']}] {paper['title'][:60]}...")

    if updated_papers:
        print(f"\n--- Updated Papers (showing first 5) ---")
        for change in updated_papers[:5]:
            print(f"  [{change['old']['id']}] upvotes: {change['old']['upvotes']} -> {change['new']['upvotes']}")

    print(f"\nRun with --update-db to apply these changes.")


def verify_papers(paper_ids: list[str]) -> None:
    """Query DB to verify papers were written correctly."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        cur = conn.cursor()

        # Get total count
        cur.execute("SELECT COUNT(*) FROM papers")
        total = cur.fetchone()[0]

        # Get count of papers we just wrote
        cur.execute("SELECT COUNT(*) FROM papers WHERE id = ANY(%s)", (paper_ids,))
        written = cur.fetchone()[0]

        # Get sample of papers we just wrote
        cur.execute(
            """
            SELECT id, title, upvotes, date
            FROM papers
            WHERE id = ANY(%s)
            ORDER BY date DESC, upvotes DESC
            LIMIT 5
            """,
            (paper_ids,),
        )
        samples = cur.fetchall()
        cur.close()

        print(f"\n{'='*60}")
        print("VERIFICATION")
        print(f"{'='*60}")
        print(f"Total papers in DB: {total}")
        print(f"Papers from this batch found in DB: {written}/{len(paper_ids)}")

        if written != len(paper_ids):
            print(f"WARNING: Expected {len(paper_ids)}, found {written}")

        print(f"\n--- Sample from DB (top 5 by date/upvotes) ---")
        for row in samples:
            print(f"  [{row[0]}] {row[1][:50]}... | upvotes: {row[2]} | date: {row[3]}")

    finally:
        conn.close()


async def main():
    parser = argparse.ArgumentParser(description="Fetch and update paper metadata from HuggingFace")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lookback-days", type=int, help="Number of days to look back from today")
    group.add_argument("--backfill", action="store_true", help="Backfill from Jan 1 of current year")
    parser.add_argument("--update-db", action="store_true", help="Actually update the database (default: preview only)")

    args = parser.parse_args()

    if not DATABASE_URL:
        print("ERROR: DATABASE_URL environment variable not set")
        return

    # Calculate lookback days
    if args.backfill:
        jan_1 = date(date.today().year, 1, 1)
        lookback_days = (date.today() - jan_1).days + 1
        print(f"Backfill mode: fetching from {jan_1} to {date.today()} ({lookback_days} days)")
    else:
        lookback_days = args.lookback_days
        print(f"Lookback mode: fetching last {lookback_days} days")

    # Fetch papers
    df = await fetch_papers_for_period(lookback_days)
    print(f"Fetched {len(df)} papers across {df['date'].nunique() if not df.empty else 0} days")

    if df.empty:
        print("No papers fetched, nothing to do.")
        return

    # Update or preview
    if args.update_db:
        print("Updating database...")
        inserted, updated = upsert_papers(df)
        print(f"Done: {inserted} inserted, {updated} updated")
        verify_papers(df["id"].tolist())
    else:
        preview_changes(df)


if __name__ == "__main__":
    asyncio.run(main())
