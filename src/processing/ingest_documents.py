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
    path: str,
    db: Any,
    metadata: Optional[Dict[str, Any]] = None,
    batch_size: int = 100,
    recursive: bool = True
) -> List[str]:
    """Ingest documents from a file or directory into the vector database.
    
    Args:
        path: Path to file or directory
        db: Database instance
        metadata: Optional metadata to attach to all documents
        batch_size: Number of documents to process in each batch
        recursive: Whether to process directories recursively
        
    Returns:
        List of document IDs
    """
    try:
        processor = DocumentProcessor()
        path = Path(path)
        
        # Handle single file
        if path.is_file():
            doc = processor.process_document(str(path), metadata)
            doc_ids = db.add_documents([doc])
            return doc_ids
        
        # Handle directory
        if not path.is_dir():
            raise ValueError(f"Path not found: {path}")
        
        # Get all files in directory
        if recursive:
            files = [f for f in path.rglob("*") if f.is_file()]
        else:
            files = [f for f in path.iterdir() if f.is_file()]
        
        # Process files in batches
        doc_ids = []
        for i in tqdm(range(0, len(files), batch_size), desc="Processing documents"):
            batch_files = files[i:i + batch_size]
            batch_docs = []
            
            for file_path in batch_files:
                try:
                    doc = processor.process_document(str(file_path), metadata)
                    batch_docs.append(doc)
                except Exception as e:
                    print(f"Warning: Could not process file {file_path}: {e}")
                    continue
            
            if batch_docs:
                batch_ids = db.add_documents(batch_docs)
                doc_ids.extend(batch_ids)
        
        return doc_ids
        
    except Exception as e:
        raise ValueError(f"Error ingesting documents: {str(e)}")

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