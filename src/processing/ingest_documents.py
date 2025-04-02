"""
Document ingestion module.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from tqdm import tqdm

from .document_processor import DocumentProcessor
from .rag_document import RAGDocument

def ingest_documents(
    file_path: str,
    db_manager: Any,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    min_chunk_size: int = 100,
    max_chunks: int = 100,
) -> List[str]:
    """
    Ingest documents from a file or directory.
    
    Args:
        file_path: Path to the file or directory to ingest
        db_manager: Database manager instance
        metadata: Optional metadata to attach to the documents
        chunk_size: Maximum number of characters per chunk
        chunk_overlap: Number of characters to overlap between chunks
        min_chunk_size: Minimum chunk size to process
        max_chunks: Maximum number of chunks per document
        
    Returns:
        List of document IDs
    """
    processor = DocumentProcessor()
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File or directory not found: {file_path}")
    
    if path.is_file():
        # Process single file
        doc = processor.process_document(str(path), metadata)
        if doc:
            doc_ids = db_manager.add_documents([doc])
            return doc_ids
        return []
    
    elif path.is_dir():
        # Process directory
        doc_ids = []
        files = list(path.rglob("*"))
        
        with tqdm(total=len(files), desc="Processing files") as pbar:
            for file in files:
                if file.is_file():
                    try:
                        doc = processor.process_document(str(file), metadata)
                        if doc:
                            ids = db_manager.add_documents([doc])
                            doc_ids.extend(ids)
                    except Exception as e:
                        print(f"Error processing {file}: {str(e)}")
                pbar.update(1)
        
        return doc_ids
    
    return []

def main():
    """Example usage of the document ingestion function."""
    # Initialize the database
    db = PostgreSQLVectorDB(
        dbname="rag_db",
        user="your_username",
        password="your_password"
    )
    
    # Example metadata
    metadata = {
        "ingestion_date": "2024-03-21",
        "source": "example_documents"
    }
    
    # Example: Ingest a single file
    file_path = "path/to/your/document.pdf"
    try:
        doc_ids = ingest_documents(file_path, db, metadata)
        print(f"Successfully ingested {len(doc_ids)} documents from {file_path}")
    except Exception as e:
        print(f"Error ingesting {file_path}: {str(e)}")
    
    # Example: Ingest a directory
    directory_path = "path/to/your/documents"
    try:
        doc_ids = ingest_documents(directory_path, db, metadata)
        print(f"Successfully ingested {len(doc_ids)} documents from {directory_path}")
    except Exception as e:
        print(f"Error ingesting {directory_path}: {str(e)}")

if __name__ == "__main__":
    main() 