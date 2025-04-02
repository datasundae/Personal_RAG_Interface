import psycopg2
from psycopg2.extras import DictCursor
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import logging

class VectorDBSearch:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.logger = logging.getLogger(__name__)
        
    def search(self, query: str, limit: int = 5, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents in the vector database"""
        try:
            # Generate query embedding
            query_embedding = self.model.encode(query)
            self.logger.info(f"Generated embedding with shape: {query_embedding.shape}")
            
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
                            filter_conditions.append(f"metadata->>'{key}' = %s")
                            params.append(str(value))
                        if filter_conditions:
                            query += " WHERE " + " AND ".join(filter_conditions)
                            self.logger.info(f"Applied metadata filter: {metadata_filter}")
                    
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
                        results.append({
                            'text': row['content'],
                            'metadata': row['metadata'],
                            'similarity': row['similarity']
                        })
                    return results
                    
        except Exception as e:
            self.logger.error(f"Error in vector search: {str(e)}")
            raise 