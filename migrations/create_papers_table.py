import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")

if not DATABASE_URL:
    raise ValueError("No database url set.")


def create_schema():
    commands = (
        """
        CREATE TABLE IF NOT EXISTS papers (
            id VARCHAR PRIMARY KEY,
            title TEXT NOT NULL,
            summary TEXT,
            authors JSONB,
            published_at TIMESTAMP,
            upvotes INTEGER DEFAULT 0,
            date DATE NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_papers_date ON papers(date)
        """,
    )
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        for command in commands:
            cur.execute(command)

        cur.close()
        conn.commit()
        print("Papers table created successfully.")
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error: {error}")
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    create_schema()
