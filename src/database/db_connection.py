"""
Database connection module.
"""

import psycopg2
from psycopg2.extras import DictCursor
import logging
from ..config.config import DB_CONFIG

logger = logging.getLogger(__name__)

class DatabaseConnection:
    _instance = None
    _connection = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._connection is None:
            self._connection = self._create_connection()

    def _create_connection(self):
        """Create a database connection."""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            conn.autocommit = True
            logger.info("Successfully connected to database")
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise

    def get_connection(self):
        """Get the database connection."""
        if self._connection is None or self._connection.closed:
            self._connection = self._create_connection()
        return self._connection

    def execute_query(self, query, params=None):
        """Execute a query and return results."""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(query, params)
                return cur.fetchall()
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise

    def execute_command(self, command, params=None):
        """Execute a command (INSERT, UPDATE, DELETE)."""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(command, params)
                conn.commit()
        except Exception as e:
            logger.error(f"Error executing command: {str(e)}")
            conn.rollback()
            raise

    def close(self):
        """Close the database connection."""
        if self._connection and not self._connection.closed:
            self._connection.close()
            self._connection = None
            logger.info("Database connection closed")

def init_db():
    """Initialize the database with required tables."""
    db = DatabaseConnection()
    conn = db.get_connection()
    cur = conn.cursor()
    
    try:
        # Create documents table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                metadata JSONB,
                embedding VECTOR(384)
            );
        """)
        
        # Create vector similarity index
        cur.execute("""
            CREATE INDEX IF NOT EXISTS documents_embedding_idx 
            ON documents 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise Exception(f"Failed to initialize database: {str(e)}")
    finally:
        cur.close()

# Create a global instance
db = DatabaseConnection()
