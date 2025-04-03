"""
Document processing module for handling various document types.
"""

from .document_processor import DocumentProcessor
from .rag_document import RAGDocument
from .ingest_documents import ingest_documents
from .pdf_processor import PDFProcessor
from .image_processor import ImageProcessor
from .book_metadata import (
    extract_metadata,
    insert_metadata,
    get_or_create_author,
    update_author_lifespan,
    merge_author_records,
    normalize_author_names,
    process_book,
    extract_text_from_pdf,
    has_multiple_authors
)

__all__ = [
    'DocumentProcessor',
    'RAGDocument',
    'ingest_documents',
    'PDFProcessor',
    'ImageProcessor',
    'extract_metadata',
    'insert_metadata',
    'get_or_create_author',
    'update_author_lifespan',
    'merge_author_records',
    'normalize_author_names',
    'process_book',
    'extract_text_from_pdf',
    'has_multiple_authors'
]

"""Document processing package.""" 
