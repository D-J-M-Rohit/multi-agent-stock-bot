import sqlite3
import hashlib

DB_NAME = "users.db"

def init_db():
    """Initialize the SQLite database and ensure the users table exists."""
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            email TEXT,
            full_name TEXT,
            password_hash TEXT
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    """Return a SHA-256 hash of the given password."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def register_user(username: str, email: str, full_name: str, password: str) -> bool:
    """
    Register a new user with the given details.
    Returns True if registration is successful, or False if the username already exists.
    """
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO users (username, email, full_name, password_hash) VALUES (?, ?, ?, ?)",
                    (username, email, full_name, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Username already exists (PRIMARY KEY constraint failed)
        return False
    finally:
        conn.close()

def verify_login(username: str, password: str) -> bool:
    """
    Verify login credentials for a username and password.
    Returns True if the credentials are valid, False otherwise.
    """
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if row:
        stored_hash = row[0]
        return stored_hash == hash_password(password)
    return False

def get_full_name(username: str) -> str:
    """Retrieve the full name of the user with the given username."""
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT full_name FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else ""
