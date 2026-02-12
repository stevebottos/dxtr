"""Migration to create user_paper_rankings and dev_user_paper_rankings tables.

These tables store paper rankings for users based on their interests.

Run with: python migrations/create_paper_rankings_table.py
"""

import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")

if not DATABASE_URL:
    raise ValueError("No database url set.")


def create_schema():
    commands = (
        # Production table
        """
        CREATE TABLE IF NOT EXISTS user_paper_rankings (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR NOT NULL,
            paper_id VARCHAR NOT NULL REFERENCES papers(id),
            paper_date DATE NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            ranking_criteria_type VARCHAR NOT NULL CHECK (ranking_criteria_type IN ('profile')),
            ranking_criteria TEXT NOT NULL,
            ranking INT NOT NULL CHECK (ranking >= 1 AND ranking <= 5),
            reason TEXT
        )
        """,
        # Lookup index
        """
        CREATE INDEX IF NOT EXISTS idx_user_paper_rankings_lookup
        ON user_paper_rankings(user_id, paper_date, ranking_criteria_type)
        """,
        # Development table (same schema)
        """
        CREATE TABLE IF NOT EXISTS dev_user_paper_rankings (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR NOT NULL,
            paper_id VARCHAR NOT NULL REFERENCES papers(id),
            paper_date DATE NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            ranking_criteria_type VARCHAR NOT NULL CHECK (ranking_criteria_type IN ('profile')),
            ranking_criteria TEXT NOT NULL,
            ranking INT NOT NULL CHECK (ranking >= 1 AND ranking <= 5),
            reason TEXT
        )
        """,
        # Lookup index - dev
        """
        CREATE INDEX IF NOT EXISTS idx_dev_user_paper_rankings_lookup
        ON dev_user_paper_rankings(user_id, paper_date, ranking_criteria_type)
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
        print("user_paper_rankings and dev_user_paper_rankings tables created successfully.")
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error: {error}")
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    create_schema()
