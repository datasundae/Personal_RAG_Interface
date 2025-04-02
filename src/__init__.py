"""
Personal RAG Interface - A modular RAG system for personal document management.

This package provides a modular system for:
1. Document processing and ingestion
2. Vector database storage and retrieval
3. Web interface for interaction
4. Configuration management
"""

<<<<<<< HEAD
from .postgres_vector_db import PostgreSQLVectorDB
from .rag_document import RAGDocument
from .app import app
=======
from .database import PostgreSQLVectorDB, RAGDocument
from .processing import ingest_documents, DocumentProcessor
from .config import MetadataManager, DB_CONFIG
from .web import app
>>>>>>> 8f39c0cbc19721b9785a7f78d10722be3f0eb339

__version__ = '0.1.0'

__all__ = [
    'PostgreSQLVectorDB',
    'RAGDocument',
<<<<<<< HEAD
    'app'
]

"""RAG AI system package.""" 
=======
    'ingest_documents',
    'DocumentProcessor',
    'MetadataManager',
    'DB_CONFIG',
    'app'
] 
>>>>>>> 8f39c0cbc19721b9785a7f78d10722be3f0eb339
