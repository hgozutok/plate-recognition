import sqlite3
from contextlib import contextmanager

DATABASE_PATH = 'plates.db'

@contextmanager
def get_db():
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Initialize the database with required tables."""
    with get_db() as conn:
        c = conn.cursor()
        
        # Create plates table
        c.execute('''
            CREATE TABLE IF NOT EXISTS plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT NOT NULL,
                capture_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                image_path TEXT NOT NULL
            )
        ''')
        
        # Create users table for authentication
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()

def add_plate_record(plate_number: str, image_path: str):
    """Add a new plate detection record."""
    with get_db() as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO plates (plate_number, image_path) VALUES (?, ?)",
            (plate_number, image_path)
        )
        conn.commit()
        return c.lastrowid

def get_plate_history():
    """Get all plate detection records."""
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM plates ORDER BY capture_time DESC")
        return c.fetchall()
