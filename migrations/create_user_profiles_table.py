import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

# Replace with your actual DATABASE_URL or environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "")

if not DATABASE_URL:
    raise ValueError("No database url set.")


def create_schema():
    commands = (
        """
        CREATE TABLE IF NOT EXISTS user_profiles (
            id VARCHAR PRIMARY KEY,
            profile TEXT NOT NULL,
            created_at DATE DEFAULT CURRENT_DATE
        )
        """,
    )
    conn = None
    try:
        # Connect to the PostgreSQL server
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        # Execute commands one by one
        for command in commands:
            cur.execute(command)

        # Close communication with the PostgreSQL database server
        cur.close()
        # Commit the changes
        conn.commit()
        print("Table created successfully.")
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error: {error}")
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    create_schema()
