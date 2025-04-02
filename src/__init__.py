"""
Personal RAG Interface - A modular RAG system for personal document management.

This package provides a modular system for:
1. Document processing and ingestion
2. Vector database storage and retrieval
3. Web interface for interaction
4. Configuration management
"""

from .postgres_vector_db import PostgreSQLVectorDB
from .rag_document import RAGDocument
from .app import app

__version__ = '0.1.0'

__all__ = [
    'PostgreSQLVectorDB',
    'RAGDocument',
    'app'
]

"""RAG AI system package.""" 