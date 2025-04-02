"""SQLite-based vector database implementation."""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

class SQLiteVectorDB:
    """SQLite-based vector database for document storage and retrieval."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the SQLite vector database.
        
        Args:
            db_path: Optional path to the SQLite database file. If not provided,
                    defaults to 'data/rag.db' in the project root.
        """
        if db_path is None:
            db_path = str(Path(__file__).parent.parent.parent / "data" / "rag.db")
        
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    async def init_db(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
            """)
            conn.commit()

    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to the database.
        
        Args:
            documents: List of document dictionaries, each containing:
                     - id: Unique document ID
                     - content: Document text content
                     - metadata: Document metadata
                     - embedding: Document vector embedding
        
        Returns:
            List of document IDs that were added.
        """
        with sqlite3.connect(self.db_path) as conn:
            for doc in documents:
                embedding_bytes = np.array(doc['embedding'], dtype=np.float32).tobytes()
                metadata_json = json.dumps(doc['metadata'])
                
                conn.execute(
                    "INSERT OR REPLACE INTO documents (id, content, metadata, embedding) VALUES (?, ?, ?, ?)",
                    (doc['id'], doc['content'], metadata_json, embedding_bytes)
                )
            conn.commit()
        
        return [doc['id'] for doc in documents]

    async def get_documents(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve documents by their IDs.
        
        Args:
            doc_ids: List of document IDs to retrieve.
        
        Returns:
            List of document dictionaries.
        """
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ','.join('?' * len(doc_ids))
            query = f"SELECT id, content, metadata, embedding FROM documents WHERE id IN ({placeholders})"
            
            results = []
            for row in conn.execute(query, doc_ids):
                doc_id, content, metadata_json, embedding_bytes = row
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32).tolist()
                metadata = json.loads(metadata_json)
                
                results.append({
                    'id': doc_id,
                    'content': content,
                    'metadata': metadata,
                    'embedding': embedding
                })
        
        return results

    async def search_documents(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Search for documents similar to the query embedding.
        
        Args:
            query_embedding: Query vector embedding.
            limit: Maximum number of results to return.
        
        Returns:
            List of document dictionaries with similarity scores.
        """
        query_array = np.array(query_embedding, dtype=np.float32)
        
        with sqlite3.connect(self.db_path) as conn:
            results = []
            for row in conn.execute("SELECT id, content, metadata, embedding FROM documents"):
                doc_id, content, metadata_json, embedding_bytes = row
                doc_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                
                # Calculate cosine similarity
                similarity = np.dot(query_array, doc_embedding) / (
                    np.linalg.norm(query_array) * np.linalg.norm(doc_embedding)
                )
                
                results.append({
                    'id': doc_id,
                    'content': content,
                    'metadata': json.loads(metadata_json),
                    'similarity': float(similarity)
                })
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]

    async def delete_documents(self, doc_ids: List[str]):
        """Delete documents from the database.
        
        Args:
            doc_ids: List of document IDs to delete.
        """
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ','.join('?' * len(doc_ids))
            conn.execute(f"DELETE FROM documents WHERE id IN ({placeholders})", doc_ids)
            conn.commit() 