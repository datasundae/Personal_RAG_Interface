"""Configuration settings for the RAG system."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_CONFIG = {
    "model": "gpt-4-turbo-preview",
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1.0
}

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Database configuration
DB_CONFIG = {
    "dbname": "musartao",
    "user": "datasundae",
    "password": "6AV%b9",
    "host": "localhost",
    "port": 5432
}

# Default metadata
DEFAULT_METADATA = {
    "ingested_by": "datasundae",
    "source": "google_docs"
}

# URL processing configuration
URL_CONFIG = {
    "timeout": 30,  # Request timeout in seconds
    "max_retries": 3,  # Maximum number of retries for failed requests
    "wait_time": 5,  # Wait time between retries in seconds
    "dynamic_wait": 10  # Wait time for dynamic content in seconds
}

# Document processing configuration
DOC_CONFIG = {
    'chunk_size': 1000,  # Number of characters per chunk
    'chunk_overlap': 200,  # Number of characters to overlap between chunks
    'min_chunk_size': 100,  # Minimum chunk size for PDF processing
    'max_chunk_size': 2000,  # Maximum chunk size for PDF processing
    'max_chunks': 100,  # Maximum number of chunks per document
    'max_file_size': 10 * 1024 * 1024,  # 10MB maximum file size
    'supported_formats': ['.txt', '.pdf'],  # Supported file formats
    'tesseract_path': '/usr/local/bin/tesseract',  # Path to Tesseract OCR executable
    'poppler_path': '/usr/local/bin/pdftoppm',  # Path to Poppler PDF tools
}

# Vector search configuration
VECTOR_CONFIG = {
    "model_name": "all-MiniLM-L6-v2",  # Sentence transformer model
    "vector_size": 384,  # Size of document vectors
    "similarity_threshold": 0.7,  # Minimum similarity score for matches
    "max_results": 10  # Maximum number of search results
}

# Hardware-specific configuration for Mac mini M4 Pro
HARDWARE_CONFIG = {
    "cpu": {
        "total_cores": 14,
        "performance_cores": 10,
        "efficiency_cores": 4,
        "memory": "64GB",
        "model": "Apple M4 Pro"
    },
    "gpu": {
        "total_cores": 20,
        "type": "Apple M4 Pro",
        "metal_support": "Metal 3",
        "vendor": "Apple"
    }
}

# GPU configuration optimized for M4 Pro
GPU_CONFIG = {
    "batch_size": 128,  # Increased batch size for M4 Pro's 14 cores
    "use_mixed_precision": True,  # Enable mixed precision for faster processing
    "num_workers": 10,  # Match number of performance cores
    "pin_memory": True,  # Pin memory for faster data transfer to GPU
    "device": "mps",  # Use Metal Performance Shaders for Apple Silicon
    "memory_fraction": 0.8,  # Use 80% of available memory (64GB)
    "prefetch_factor": 2,  # Prefetch factor for data loading
    "persistent_workers": True,  # Keep workers alive between batches
    "use_amp": True  # Use Automatic Mixed Precision
}

# Model configuration optimized for M4 Pro
MODEL_CONFIG = {
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',  # Model for generating embeddings
    'embedding_dimension': 384,  # Dimension of the embeddings
    'batch_size': 32,  # Batch size for processing documents
    'device': 'mps',  # Use Metal Performance Shaders
    'num_workers': 10,  # Match number of performance cores
    'use_amp': True,  # Use Automatic Mixed Precision
    'pin_memory': True  # Pin memory for faster data transfer
}

# Cache configuration
CACHE_CONFIG = {
    'cache_dir': str(PROJECT_ROOT / "cache"),  # Directory for caching embeddings and other data
    'max_cache_size': 1024 * 1024 * 1024,  # 1GB maximum cache size
    'cache_ttl': 86400,  # Cache time-to-live in seconds (24 hours)
}

# Create necessary directories
for directory in [
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "cache",
]:
    directory.mkdir(parents=True, exist_ok=True)
