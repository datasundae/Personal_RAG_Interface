<<<<<<< HEAD
"""
Document ingestion module.
"""

=======
>>>>>>> 8f39c0cbc19721b9785a7f78d10722be3f0eb339
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from tqdm import tqdm

<<<<<<< HEAD
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
=======
from src.processing.document_processor import DocumentProcessor
from src.database.postgres_vector_db import PostgreSQLVectorDB
from src.processing.rag_document import RAGDocument

def ingest_documents(
    path: str,
    db: PostgreSQLVectorDB,
    metadata: Optional[Dict[str, Any]] = None,
    batch_size: int = 100,
    recursive: bool = True
) -> List[str]:
    """Ingest documents from a file or directory into the vector database.
    
    Args:
        path: Path to file or directory
        db: PostgreSQL vector database instance
        metadata: Optional metadata to attach to all documents
        batch_size: Number of documents to process in each batch
        recursive: Whether to process directories recursively
>>>>>>> 8f39c0cbc19721b9785a7f78d10722be3f0eb339
        
    Returns:
        List of document IDs
    """
<<<<<<< HEAD
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
=======
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
>>>>>>> 8f39c0cbc19721b9785a7f78d10722be3f0eb339

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