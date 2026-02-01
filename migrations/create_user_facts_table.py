"""Migration to create user_facts and dev_user_facts tables.

These tables store facts learned about users through conversation.
Facts are stored with timestamps to maintain chronological order.

Run with: python migrations/create_user_facts_table.py
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
        CREATE TABLE IF NOT EXISTS user_facts (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR NOT NULL,
            fact TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_user_facts_user_id
        ON user_facts(user_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_user_facts_created_at
        ON user_facts(created_at)
        """,
        # Development table (same schema)
        """
        CREATE TABLE IF NOT EXISTS dev_user_facts (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR NOT NULL,
            fact TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_dev_user_facts_user_id
        ON dev_user_facts(user_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_dev_user_facts_created_at
        ON dev_user_facts(created_at)
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
        print("user_facts and dev_user_facts tables created successfully.")
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error: {error}")
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    create_schema()
