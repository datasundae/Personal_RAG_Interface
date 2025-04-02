# Personal RAG Interface - Project Context

## Project Overview
A modular RAG (Retrieval-Augmented Generation) system designed for personal document management, with a focus on maintaining clean separation of concerns and modularity.

## Core Components

### 1. Database Module (`src/database/`)
- **Purpose**: Vector storage and retrieval operations
- **Key Components**:
  - `postgres_vector_db.py`: PostgreSQL vector database operations
  - `rag_document.py`: Document model and structure
- **Responsibilities**:
  - Document storage with encryption
  - Vector similarity search
  - Database schema management
  - Connection handling

### 2. Processing Module (`src/processing/`)
- **Purpose**: Document processing pipeline
- **Key Components**:
  - `document_processor.py`: Base document processor
  - `pdf_processor.py`: PDF-specific processing
  - `image_processor.py`: Image-specific processing
  - `ingest_documents.py`: Document ingestion interface
- **Responsibilities**:
  - Document type detection
  - Content extraction
  - Metadata processing
  - Chunking and normalization

### 3. Configuration Module (`src/config/`)
- **Purpose**: Application configuration management
- **Key Components**:
  - `config.py`: General configuration settings
  - `metadata_config.py`: Metadata management
- **Responsibilities**:
  - Database configuration
  - Document processing settings
  - Metadata templates
  - Default values

### 4. Web Interface (`src/web/`)
- **Purpose**: User interaction and document management
- **Key Components**:
  - `app.py`: Flask application
- **Responsibilities**:
  - Document upload
  - Search interface
  - Metadata management
  - Result display

## Data Flow
1. User uploads document through web interface
2. Document processor identifies type and extracts content
3. Content is processed and normalized
4. Document is stored in vector database
5. User can search and retrieve documents

## Security Considerations
- Sensitive data encryption in database
- Secure file handling
- Input validation
- Access control

## Dependencies
- PostgreSQL with pgvector extension
- Python 3.8+
- Various document processing libraries
- Web framework (Flask)
- Vector embedding models

## Development Guidelines
- Maintain modularity between components
- Keep document processing independent
- Follow clean code principles
- Document all major functions
- Include type hints
- Write unit tests for each module 