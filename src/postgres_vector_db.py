import os
import psycopg2
from psycopg2.extras import DictCursor
import numpy as np
from typing import List, Dict, Any, Optional
from .rag_document import RAGDocument
import logging

class PostgreSQLVectorDB:
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize the PostgreSQL vector database connection.
        
        Args:
            connection_string: Optional connection string. If not provided,
                            will use DATABASE_URL from environment variables.
        """
        self.logger = logging.getLogger(__name__)
        self.connection_string = connection_string or os.environ.get('DATABASE_URL', 'postgresql://datasundae:6AV%25b9@localhost:5432/musartao')
        if not self.connection_string:
            raise ValueError("Database connection string not provided")
            
        self._init_db()
    
    def _init_db(self):
        """Initialize the database schema."""
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                # Create extension for vector operations
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Create documents table if it doesn't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id SERIAL PRIMARY KEY,
                        content TEXT NOT NULL,
                        embedding vector(384),
                        metadata JSONB,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create index for similarity search
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                    ON documents 
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)
                
                conn.commit()
    
    def add_document(self, document: RAGDocument, embedding: np.ndarray):
        """Add a document to the database.
        
        Args:
            document: RAGDocument object containing text and metadata
            embedding: Vector embedding of the document text
        """
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO documents (content, embedding, metadata)
                    VALUES (%s, %s, %s)
                    RETURNING id;
                """, (document.text, embedding.tolist(), document.metadata))
                conn.commit()
                return cur.fetchone()[0]
    
    def search(self, query_embedding: np.ndarray, limit: int = 5, metadata_filter: Optional[Dict[str, Any]] = None) -> List[RAGDocument]:
        """Search for similar documents.
        
        Args:
            query_embedding: Vector embedding of the query
            limit: Maximum number of results to return
            metadata_filter: Optional metadata filter with support for multiple conditions
            
        Returns:
            List of RAGDocument objects
        """
        self.logger.info(f"Searching vector database with embedding of shape {query_embedding.shape}")
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    # Construct base query
                    query = """
                        SELECT content, metadata, 1 - (embedding <=> %s::vector) as similarity
                        FROM documents
                    """
                    params = [query_embedding.tolist()]
                    
                    # Add metadata filter if provided
                    if metadata_filter:
                        filter_conditions = []
                        for key, value in metadata_filter.items():
                            # Handle simple equality condition
                            filter_conditions.append(f"metadata->>'{key}' = %s")
                            params.append(str(value))
                        if filter_conditions:
                            query += " WHERE " + " AND ".join(filter_conditions)
                            self.logger.info(f"Applied metadata filter: {metadata_filter}")
                            self.logger.info(f"Generated SQL: {query}")
                            self.logger.info(f"With parameters: {params}")
                    
                    # Add ordering and limit
                    query += """
                        ORDER BY similarity DESC
                        LIMIT %s;
                    """
                    params.append(limit)
                    
                    cur.execute(query, params)
                    
                    results = []
                    for row in cur.fetchall():
                        self.logger.info(f"Found document with similarity: {row['similarity']}")
                        self.logger.info(f"Document metadata: {row['metadata']}")
                        results.append(RAGDocument(
                            text=row['content'],
                            metadata=row['metadata']
                        ))
                    self.logger.info(f"Returning {len(results)} documents from vector search")
                    return results
        except Exception as e:
            self.logger.error(f"Error in vector search: {str(e)}")
            raise
    
    def delete_document(self, document_id: int):
        """Delete a document from the database.
        
        Args:
            document_id: ID of the document to delete
        """
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM documents WHERE id = %s;", (document_id,))
                conn.commit()
    
    def get_document(self, document_id: int) -> Optional[RAGDocument]:
        """Retrieve a document by ID.
        
        Args:
            document_id: ID of the document to retrieve
            
        Returns:
            RAGDocument object if found, None otherwise
        """
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute("""
                    SELECT content, metadata
                    FROM documents
                    WHERE id = %s;
                """, (document_id,))
                
                row = cur.fetchone()
                if row:
                    return RAGDocument(
                        text=row['content'],
                        metadata=row['metadata']
                    )
                return None 