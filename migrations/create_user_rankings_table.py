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
        CREATE TABLE IF NOT EXISTS user_rankings (
            user_id VARCHAR NOT NULL,
            date VARCHAR NOT NULL,
            rankings JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            expires_at TIMESTAMP DEFAULT NOW() + INTERVAL '24 hours',
            PRIMARY KEY (user_id, date)
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_user_rankings_expires
        ON user_rankings(expires_at)
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
        print("User rankings table created successfully.")
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error: {error}")
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    create_schema()
