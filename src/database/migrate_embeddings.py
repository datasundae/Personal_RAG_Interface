"""
Script to migrate documents to the new embedding model.
"""

import psycopg2
from sentence_transformers import SentenceTransformer
import logging
from tqdm import tqdm
import time
import json
from psycopg2.extras import Json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection parameters
DB_PARAMS = {
    "dbname": "musartao",
    "user": "datasundae",
    "password": "6AV%b9",
    "host": "localhost",
    "port": 5432
}

def migrate_embeddings(batch_size=100):
    """Migrate documents to the new embedding model."""
    try:
        # Initialize the new model
        logger.info("Loading new embedding model...")
        new_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        # Connect to the database
        logger.info("Connecting to database...")
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        
        # Get total count of documents
        cur.execute("SELECT COUNT(*) FROM documents_backup")
        total_docs = cur.fetchone()[0]
        logger.info(f"Total documents to migrate: {total_docs}")
        
        # Process documents in batches
        offset = 0
        while True:
            # Fetch a batch of documents
            cur.execute("""
                SELECT id, content, metadata, encrypted_content
                FROM documents_backup
                ORDER BY id
                LIMIT %s OFFSET %s
            """, (batch_size, offset))
            
            batch = cur.fetchall()
            if not batch:
                break
                
            logger.info(f"Processing batch {offset//batch_size + 1}...")
            
            # Process each document
            for doc_id, content, metadata, encrypted_content in tqdm(batch):
                try:
                    # Generate new embedding
                    new_embedding = new_model.encode(content)
                    
                    # Ensure metadata is properly formatted
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except json.JSONDecodeError:
                            metadata = {}
                    elif metadata is None:
                        metadata = {}
                    
                    # Insert into new table with proper JSON handling
                    cur.execute("""
                        INSERT INTO documents_new (id, content, metadata, encrypted_content, embedding)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        doc_id,
                        content,
                        Json(metadata),  # Use psycopg2's Json adapter
                        encrypted_content,
                        new_embedding.tolist()
                    ))
                    
                except Exception as e:
                    logger.error(f"Error processing document {doc_id}: {str(e)}")
                    logger.error(f"Metadata type: {type(metadata)}")
                    logger.error(f"Metadata content: {metadata}")
                    continue
            
            # Commit after each batch
            conn.commit()
            offset += batch_size
            
            # Add a small delay to prevent overwhelming the system
            time.sleep(1)
        
        logger.info("Migration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during migration: {str(e)}")
        raise
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    migrate_embeddings() 