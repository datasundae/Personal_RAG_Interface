import os
import zipfile
from typing import Optional, Dict, List
import psycopg2
from bs4 import BeautifulSoup
import re
from datetime import datetime
import argparse
from src.embeddings import create_embeddings_and_save
from src.rag_document import RAGDocument
import json
import tiktoken
from pathlib import Path
from tqdm import tqdm

class GutenbergProcessor:
    def __init__(self, db_params: Dict[str, str]):
        """Initialize Gutenberg processor with database connection parameters."""
        self.db_params = db_params
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
    def process_text_file(self, file_path: str) -> str:
        """Process a plain text file from Project Gutenberg.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Cleaned text content
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Remove Project Gutenberg header and footer
        text = re.sub(r'^.*?START OF (?:THIS|THE) PROJECT GUTENBERG.*?\n', '', text, flags=re.DOTALL)
        text = re.sub(r'END OF (?:THIS|THE) PROJECT GUTENBERG.*$', '', text, flags=re.DOTALL)
        
        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = text.strip()
        
        return text
        
    def process_html_file(self, file_path: str) -> str:
        """Process an HTML file from Project Gutenberg.
        
        Args:
            file_path: Path to the HTML file
            
        Returns:
            Cleaned text content
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            html = f.read()
            
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove Project Gutenberg header and footer
        for element in soup.find_all(['div', 'p']):
            if 'gutenberg' in element.get_text().lower():
                element.decompose()
                
        # Extract main content
        text = soup.get_text()
        
        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = text.strip()
        
        return text
        
    def process_zip_file(self, zip_path: str) -> str:
        """Process a zipped collection of HTML files.
        
        Args:
            zip_path: Path to the zip file
            
        Returns:
            Combined cleaned text content
        """
        text_parts = []
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.endswith('.html'):
                    with zip_ref.open(file_name) as f:
                        html = f.read().decode('utf-8')
                        text = self.process_html_file(html)
                        text_parts.append(text)
                        
        return '\n\n'.join(text_parts)
        
    def extract_metadata(self, text: str, filename: str, filepath: str) -> Dict[str, str]:
        """Extract metadata from text and filename.
        
        Args:
            text: Extracted text
            filename: Name of the file
            filepath: Full path to the file
            
        Returns:
            Dictionary containing extracted metadata
        """
        print(f"\nExtracting metadata for {filename}")
        
        metadata = {
            'filename': filename,
            'file_name': filename,
            'file_path': filepath,
            'title': None,
            'author': None,
            'publication_date': None,
            'publisher': None,
            'isbn': None,
            'language': None,
            'description': None,
            'thumbnail': None,
            'page_count': None,
            'created_at': datetime.now().isoformat()
        }
        
        # Try to extract title and author from filename
        name_without_ext = os.path.splitext(filename)[0]
        parts = re.split(r'[-_–—]', name_without_ext)
        if len(parts) >= 2:
            metadata['author'] = parts[0].strip()
            metadata['title'] = ' '.join(parts[1:]).strip()
        else:
            metadata['title'] = name_without_ext.replace('_', ' ').replace('-', ' ').strip()
            
        # Try to extract additional metadata from the text
        first_lines = text.split('\n')[:20]
        
        # Look for title in first few lines
        if not metadata['title']:
            for line in first_lines:
                line = line.strip()
                if len(line) > 10 and not any(re.search(pattern, line.lower()) for pattern in [
                    r'copyright', r'all rights reserved', r'page', r'table of contents',
                    r'contents', r'preface', r'introduction', r'foreword', r'chapter'
                ]):
                    metadata['title'] = line
                    break
                    
        # Look for author in first few lines
        if not metadata['author']:
            for line in first_lines:
                line = line.strip()
                if len(line) > 5 and not any(re.search(pattern, line.lower()) for pattern in [
                    r'copyright', r'all rights reserved', r'page', r'table of contents',
                    r'contents', r'preface', r'introduction', r'foreword', r'chapter'
                ]):
                    metadata['author'] = line
                    break
                    
        # Try to find publication date
        date_patterns = [
            r'©\s*(\d{4})',
            r'published\s+in\s+(\d{4})',
            r'first published\s+(\d{4})',
            r'copyright\s+(\d{4})',
            r'edition\s+(\d{4})',
            r'printing\s+(\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                year = int(match.group(1))
                if 1800 <= year <= datetime.now().year:
                    metadata['publication_date'] = str(year)
                    break
                    
        # Try to find publisher
        publisher_patterns = [
            r'published by\s+([A-Za-z\s&]+(?:\s+[A-Za-z]+)*)',
            r'publisher:\s*([A-Za-z\s&]+(?:\s+[A-Za-z]+)*)',
            r'©\s*\d{4}\s*([A-Za-z\s&]+(?:\s+[A-Za-z]+)*)'
        ]
        
        for pattern in publisher_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                publisher = match.group(1).strip()
                if len(publisher) > 3:
                    metadata['publisher'] = publisher
                    break
                    
        # Set language based on filename or content
        if any(word in filename.lower() for word in ['deutsch', 'german', 'deu']):
            metadata['language'] = 'German'
        elif any(word in filename.lower() for word in ['francais', 'french', 'fra']):
            metadata['language'] = 'French'
        else:
            metadata['language'] = 'English'
            
        # Calculate approximate page count (assuming 2500 characters per page)
        metadata['page_count'] = len(text) // 2500 + 1
        
        return metadata
        
    def process_file(self, file_path: str):
        """Process a Project Gutenberg file (text, HTML, or zip).
        
        Args:
            file_path: Path to the file to process
        """
        print(f"\nProcessing book: {file_path}")
        
        # Determine file type and process accordingly
        if file_path.endswith('.txt'):
            text = self.process_text_file(file_path)
        elif file_path.endswith('.html'):
            text = self.process_html_file(file_path)
        elif file_path.endswith('.zip'):
            text = self.process_zip_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
            
        # Extract metadata
        filename = os.path.basename(file_path)
        metadata = self.extract_metadata(text, filename, file_path)
        
        # Update database
        print("\nUpdating database with metadata...")
        try:
            conn = psycopg2.connect(**self.db_params)
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO books (filename, file_name, file_path, title, author, publication_date, publisher, isbn, language, description, thumbnail, page_count, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (filename) 
                    DO UPDATE SET
                        file_name = EXCLUDED.file_name,
                        file_path = EXCLUDED.file_path,
                        title = EXCLUDED.title,
                        author = EXCLUDED.author,
                        publication_date = EXCLUDED.publication_date,
                        publisher = EXCLUDED.publisher,
                        isbn = EXCLUDED.isbn,
                        language = EXCLUDED.language,
                        description = EXCLUDED.description,
                        thumbnail = EXCLUDED.thumbnail,
                        page_count = EXCLUDED.page_count,
                        created_at = EXCLUDED.created_at
                """, (
                    metadata['filename'],
                    metadata['file_name'],
                    metadata['file_path'],
                    metadata['title'],
                    metadata['author'],
                    metadata['publication_date'],
                    metadata['publisher'],
                    metadata['isbn'],
                    metadata['language'],
                    metadata['description'],
                    metadata['thumbnail'],
                    metadata['page_count'],
                    metadata['created_at']
                ))
                conn.commit()
                print("Database updated successfully")
        except Exception as e:
            print(f"Error updating database: {str(e)}")
        finally:
            if 'conn' in locals():
                conn.close()
                
        # Create text chunks
        print("\nCreating text chunks...")
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_size = 1000  # characters per chunk
        chunk_overlap = 200  # characters overlap between chunks
        
        for paragraph in text.split('\n\n'):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            if current_length + len(paragraph) > chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_length = len(paragraph)
            else:
                current_chunk.append(paragraph)
                current_length += len(paragraph)
                
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            
        print(f"Created {len(chunks)} chunks")
        
        # Create RAG documents
        print("\nCreating RAG documents...")
        rag_documents = []
        for i, chunk in enumerate(chunks):
            doc = RAGDocument(
                text=chunk,
                metadata={
                    **metadata,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'source': file_path,
                    'character_count': len(chunk),
                    'token_count': len(self.encoding.encode(chunk))
                }
            )
            rag_documents.append(doc)
            
        # Create embeddings
        print("\nCreating embeddings...")
        create_embeddings_and_save(rag_documents, metadata['filename'])
        
        print(f"Successfully processed and updated database for {filename}")

def main():
    parser = argparse.ArgumentParser(description='Process Project Gutenberg texts')
    parser.add_argument('file_path', help='Path to the text/HTML/zip file to process')
    args = parser.parse_args()
    
    # Database connection parameters
    db_params = {
        'dbname': 'musartao',
        'user': 'datasundae',
        'password': '6AV%b9',
        'host': 'localhost',
        'port': '5432'
    }
    
    processor = GutenbergProcessor(db_params)
    processor.process_file(args.file_path)

if __name__ == '__main__':
    main() 