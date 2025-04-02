"""
Database connection module.
"""

import psycopg2
import psycopg2.extras
from typing import Optional, Dict, Any

from ..config.config import DB_CONFIG

def get_db_connection(config: Optional[Dict[str, Any]] = None):
    """Get a database connection."""
    if config is None:
        config = DB_CONFIG
    
    try:
        conn = psycopg2.connect(**config)
        return conn
    except Exception as e:
        raise ConnectionError(f"Failed to connect to database: {str(e)}")

def init_db():
    """Initialize the database with required tables."""
    conn = get_db_connection()
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
        conn.close() 