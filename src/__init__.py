"""
Personal RAG Interface - A modular RAG system for personal document management.

This package provides a modular system for:
1. Document processing and ingestion
2. Vector database storage and retrieval
3. Web interface for interaction
4. Configuration management
"""

from .database import PostgreSQLVectorDB, RAGDocument
from .processing import ingest_documents, DocumentProcessor
from .config import MetadataManager, DB_CONFIG
from .web import app

__version__ = '0.1.0'

__all__ = [
    'PostgreSQLVectorDB',
    'RAGDocument',
    'ingest_documents',
    'DocumentProcessor',
    'MetadataManager',
    'DB_CONFIG',
    'app'
]

"""RAG AI system package.""" 
