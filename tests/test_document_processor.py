"""Tests for the document processor module."""

import os
import pytest
from pathlib import Path
import tempfile
import shutil

from src.processing.document_processor import DocumentProcessor
from src.database.sqlite_vector_db import SQLiteVectorDB

@pytest.fixture(scope="function")
def db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        yield str(db_path)

@pytest.fixture(scope="function")
async def db(db_path):
    """Create a temporary database for testing."""
    db = SQLiteVectorDB(db_path)
    await db.init_db()
    return db

@pytest.fixture(scope="function")
async def processor(db):
    """Create a document processor instance for testing."""
    db_instance = await db
    return DocumentProcessor(db_instance)

@pytest.fixture(scope="function")
def test_files():
    """Create temporary test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a text file
        text_path = Path(temp_dir) / "test.txt"
        with open(text_path, "w") as f:
            f.write("This is a test document.\nIt has multiple lines.\nAnd some content to process.")
        
        # Create a PDF file (simple text-based)
        pdf_path = Path(temp_dir) / "test.pdf"
        # TODO: Add PDF creation logic when needed
        
        yield {
            "text": str(text_path),
            "pdf": str(pdf_path),
            "dir": temp_dir
        }

@pytest.mark.asyncio
async def test_process_text_file(processor, test_files):
    """Test processing a text file."""
    processor_instance = await processor
    doc_ids = await processor_instance.process_file(test_files["text"])
    assert len(doc_ids) > 0
    
    # Test search functionality
    results = await processor_instance.search("test document")
    assert len(results) > 0
    assert "test document" in results[0]["content"].lower()

@pytest.mark.asyncio
async def test_process_directory(processor, test_files):
    """Test processing a directory of files."""
    processor_instance = await processor
    doc_ids = await processor_instance.process_directory(test_files["dir"])
    assert len(doc_ids) > 0
    
    # Test search across all documents
    results = await processor_instance.search("multiple lines")
    assert len(results) > 0
    assert any("multiple lines" in result["content"].lower() for result in results)

@pytest.mark.asyncio
async def test_invalid_file(processor):
    """Test handling of invalid files."""
    processor_instance = await processor
    with tempfile.NamedTemporaryFile(suffix=".invalid") as temp_file:
        with pytest.raises(ValueError):
            await processor_instance.process_file(temp_file.name)

@pytest.mark.asyncio
async def test_chunking(processor, test_files):
    """Test text chunking functionality."""
    processor_instance = await processor
    # Create a long text file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as temp_file:
        long_text = " ".join(["word"] * 1000)  # Create text longer than chunk_size
        temp_file.write(long_text)
        temp_file.flush()
        
        doc_ids = await processor_instance.process_file(temp_file.name)
        assert len(doc_ids) > 1  # Should create multiple chunks

@pytest.mark.asyncio
async def test_metadata(processor, test_files):
    """Test metadata handling."""
    processor_instance = await processor
    metadata = {"source": "test", "category": "unit_test"}
    doc_ids = await processor_instance.process_file(test_files["text"], metadata=metadata)
    
    # Verify metadata in search results
    results = await processor_instance.search("test document")
    assert len(results) > 0
    result_metadata = results[0]["metadata"]
    assert result_metadata["source"] == "test"
    assert result_metadata["category"] == "unit_test"
    assert "source_file" in result_metadata
    assert "chunk_index" in result_metadata 