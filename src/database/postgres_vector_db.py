<<<<<<< HEAD
"""
PostgreSQL vector database implementation.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import psycopg2
import psycopg2.extras
import json
import logging
from typing import List, Tuple, Optional, Dict, Any

from ..processing.rag_document import RAGDocument
from .db_connection import get_db_connection, init_db
=======
from typing import List, Optional, Dict, Any, Tuple
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import json
from cryptography.fernet import Fernet
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .rag_document import RAGDocument
>>>>>>> 8f39c0cbc19721b9785a7f78d10722be3f0eb339

class PostgreSQLVectorDB:
    """PostgreSQL vector database with encryption for sensitive data."""
    
    def __init__(
        self,
        dbname: str = "musartao",
        user: str = "datasundae",
        password: str = "6AV%b9",
        host: str = "localhost",
        port: int = 5432
    ):
        """Initialize the PostgreSQL vector database.
        
        Args:
            dbname: Database name
            user: Database user
            password: Database password
            host: Database host
            port: Database port
        """
        self.conn_params = {
            "dbname": dbname,
            "user": user,
            "password": password,
            "host": host,
            "port": port
        }
        
        # Initialize encryption
        self._init_encryption()
        
        # Initialize sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create tables if they don't exist
        self._init_db()
    
    def _init_encryption(self):
        """Initialize encryption key and Fernet instance."""
        # Generate a key from a password (in production, use a secure key management system)
        password = os.getenv('DB_ENCRYPTION_KEY', 'your-secure-password-here')
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        self.fernet = Fernet(key)
    
    def _encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.fernet.encrypt(data.encode()).decode()
    
    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    def _init_db(self):
        """Initialize the database with encrypted columns."""
        try:
            with psycopg2.connect(**self.conn_params) as conn:
                with conn.cursor() as cur:
                    # Enable pgvector extension
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    
                    # Check if the table exists and has the required columns
                    cur.execute("""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = 'documents';
                    """)
                    existing_columns = [row[0] for row in cur.fetchall()]
                    
                    # If table exists but missing required columns, drop it
                    if existing_columns and 'encrypted_content' not in existing_columns:
                        cur.execute("DROP TABLE documents;")
                        conn.commit()
                    
                    # Create documents table with encrypted columns
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS documents (
                            id SERIAL PRIMARY KEY,
                            content TEXT NOT NULL,
                            encrypted_content TEXT NOT NULL,
                            metadata JSONB,
                            embedding vector(384)
                        );
                    """)
                    
                    # Create index on embedding
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                        ON documents 
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                    """)
                    
                    conn.commit()
                    
        except Exception as e:
            raise ValueError(f"Error initializing database: {str(e)}")
    
    def add_documents(self, documents: List[RAGDocument]) -> List[str]:
        """Add documents to the database with encryption.
        
        Args:
            documents: List of RAGDocument instances
            
        Returns:
            List of document IDs
        """
        try:
            with psycopg2.connect(**self.conn_params) as conn:
                with conn.cursor() as cur:
                    # Prepare data for batch insert
                    data = []
                    for doc in documents:
                        # Generate embedding
                        embedding = self.model.encode(doc.text)
                        
                        # Encrypt sensitive content
                        encrypted_content = self._encrypt_data(doc.text)
                        
                        # Prepare metadata
                        metadata = json.dumps(doc.metadata)
                        
                        data.append((
                            doc.text,  # Keep original content for embedding
                            encrypted_content,
                            metadata,
                            embedding.tolist()
                        ))
                    
                    # Batch insert documents
                    query = """
                        INSERT INTO documents (content, encrypted_content, metadata, embedding)
                        VALUES %s
                        RETURNING id;
                    """
                    template = "(%s, %s, %s, %s::vector)"
                    
                    doc_ids = []
                    results = execute_values(cur, query, data, template=template, fetch=True)
                    for result in results:
                        doc_ids.append(str(result[0]))
                    
                    conn.commit()
                    
                    return doc_ids
                    
        except Exception as e:
            raise ValueError(f"Error adding documents: {str(e)}")
    
    def search(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[RAGDocument, float]]:
        """Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            metadata_filter: Optional metadata filter with support for multiple conditions
            
        Returns:
            List of tuples (RAGDocument, similarity)
        """
        try:
            # Generate query embedding
            query_embedding = self.model.encode(query)
            
            with psycopg2.connect(**self.conn_params) as conn:
                with conn.cursor() as cur:
                    # Construct SQL query
                    sql = """
                        SELECT id, content, encrypted_content, metadata, 
                               1 - (embedding <=> %s::vector) as similarity
                        FROM documents
                    """
                    params = [query_embedding.tolist()]
                    
                    # Add metadata filter if provided
                    if metadata_filter:
                        filter_sql, filter_params = self._format_metadata_filter(metadata_filter)
                        sql += " WHERE " + filter_sql
                        params.extend(filter_params)
                    
                    # Add similarity threshold and limit
                    sql += """
                        ORDER BY similarity DESC
                        LIMIT %s;
                    """
                    params.append(k)
                    
                    # Execute query
                    cur.execute(sql, params)
                    results = cur.fetchall()
                    
                    # Format results
                    documents = []
                    for row in results:
                        doc_id, content, encrypted_content, metadata, similarity = row
                        doc = RAGDocument(
                            text=content,
                            metadata=metadata
                        )
                        documents.append((doc, float(similarity)))
                    
                    return documents
                    
        except Exception as e:
            raise ValueError(f"Error searching documents: {str(e)}")
    
    def _format_metadata_filter(self, metadata_filter: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """Format metadata filter into SQL condition and parameters.
        
        Args:
            metadata_filter: Dictionary of metadata key-value pairs or conditions
            
        Returns:
            Tuple of (SQL condition string, list of parameters)
        """
        conditions = []
        params = []
        
        for key, value in metadata_filter.items():
            if isinstance(value, dict):
                # Handle complex conditions (e.g., {'$in': ['value1', 'value2']})
                for op, val in value.items():
                    if op == '$in':
                        placeholders = ','.join(['%s'] * len(val))
                        conditions.append(f"metadata->>'{key}' IN ({placeholders})")
                        params.extend(val)
                    elif op == '$regex':
                        conditions.append(f"metadata->>'{key}' ~ %s")
                        params.append(val)
                    elif op == '$exists':
                        if val:
                            conditions.append(f"metadata ? '{key}'")
                        else:
                            conditions.append(f"NOT metadata ? '{key}'")
            else:
                # Handle simple equality condition
                conditions.append(f"metadata->>'{key}' = %s")
                params.append(str(value))
        
        return " AND ".join(conditions), params
    
    def get_document(self, doc_id: str) -> Optional[RAGDocument]:
        """Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            RAGDocument instance if found, None otherwise
        """
        try:
            with psycopg2.connect(**self.conn_params) as conn:
                with conn.cursor() as cur:
                    # Query document
                    cur.execute("""
                        SELECT content, metadata, embedding
                        FROM documents
                        WHERE id = %s;
                    """, [doc_id])
                    
                    row = cur.fetchone()
                    if row is None:
                        return None
                    
                    content, metadata, embedding = row
                    return RAGDocument(
                        text=content,
                        metadata=metadata,
                        embedding=np.array(embedding)
                    )
                    
        except Exception as e:
            raise ValueError(f"Error getting document: {str(e)}")
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if document was deleted, False otherwise
        """
        try:
            with psycopg2.connect(**self.conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM documents WHERE id = %s;", [doc_id])
                    deleted = cur.rowcount > 0
                    conn.commit()
                    return deleted
                    
        except Exception as e:
            raise ValueError(f"Error deleting document: {str(e)}")
    
    def clear(self):
        """Clear all documents from the database."""
        try:
            with psycopg2.connect(**self.conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("TRUNCATE TABLE documents;")
                    conn.commit()
                    
        except Exception as e:
            raise ValueError(f"Error clearing database: {str(e)}")
    
    def close(self):
        """Close any open resources."""
        pass  # Connection is handled by context managers
<<<<<<< HEAD
    
    def get_books(self) -> List[Dict[str, str]]:
        """Get the list of books from the documents table.
        
        Returns:
            List of dictionaries containing book titles and authors.
        """
        try:
            with psycopg2.connect(**self.conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT metadata->>'title' as title, metadata->>'author' as author
                        FROM documents
                        WHERE metadata->>'type' = 'book';
                    """)
                    books = [{'title': row[0], 'author': row[1]} for row in cur.fetchall()]
                    return books
        except Exception as e:
            raise ValueError(f"Error retrieving books: {str(e)}")
=======
>>>>>>> 8f39c0cbc19721b9785a7f78d10722be3f0eb339

# Example usage
if __name__ == "__main__":
    # Initialize database
    db = PostgreSQLVectorDB(
        dbname="musartao",
        user="datasundae",
        password="6AV%b9"
    )
    
    # Add sample documents
    documents = [
        RAGDocument(
            content="PostgreSQL with pgvector is a powerful combination for vector similarity search.",
            metadata={"source": "example1", "type": "test"}
        ),
        RAGDocument(
            content="Vector databases are essential for semantic search and recommendation systems.",
            metadata={"source": "example2", "type": "test"}
        )
    ]
    
    doc_ids = db.add_documents(documents)
    print(f"Added documents with IDs: {doc_ids}")
    
    # Search for similar documents
    results = db.search("what is vector similarity search")
    print("\nSearch results:")
    for doc, similarity in results:
        print(f"Content: {doc.content}")
        print(f"Metadata: {doc.metadata}")
        print(f"Similarity: {similarity}")
        print("-" * 80) 