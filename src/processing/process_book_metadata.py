import os
import pdfplumber
import PyPDF2
import psycopg2
from datetime import datetime
from pathlib import Path
import re
import subprocess
from io import StringIO
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
import sys

def get_pdf_info(pdf_path):
    """Get PDF metadata using pdfinfo command."""
    try:
        result = subprocess.run(['pdfinfo', pdf_path], capture_output=True, text=True)
        if result.returncode == 0:
            info = {}
            for line in result.stdout.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    info[key.strip()] = value.strip()
            return info
    except Exception as e:
        print(f"Error running pdfinfo: {str(e)}")
    return None

def extract_text_from_pdf(pdf_path, max_pages=None):
    """Extract text from PDF using multiple methods."""
    text = ""
    
    # Try pdftotext first as it's usually the fastest
    try:
        process = subprocess.run(['pdftotext', 
                                '-f', '1', 
                                '-l', str(max_pages if max_pages else 4),
                                pdf_path, '-'], 
                                capture_output=True, 
                                text=True)
        if process.stdout.strip():
            return process.stdout
    except Exception as e:
        print(f"Error with pdftotext: {e}")

    # Try pdfplumber if pdftotext fails
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages_to_process = pdf.pages[:max_pages] if max_pages else pdf.pages[:4]
            for page in pages_to_process:
                try:
                    text += page.extract_text() + "\n"
                except Exception as e:
                    print(f"Error extracting text from page with pdfplumber: {e}")
        if text.strip():
            return text
    except Exception as e:
        print(f"Error with pdfplumber: {e}")

    # Try PyPDF2 as last resort
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = min(len(reader.pages), max_pages if max_pages else 4)
            for page_num in range(num_pages):
                try:
                    text += reader.pages[page_num].extract_text() + "\n"
                except Exception as e:
                    print(f"Error extracting text from page with PyPDF2: {e}")
        if text.strip():
            return text
    except Exception as e:
        print(f"Error with PyPDF2: {e}")

    return text

def has_multiple_authors(author):
    """Check if the author string contains multiple authors."""
    if not author:
        return False
        
    # Clean the author string
    author = author.lower().strip()
    
    # Remove common phrases that might indicate multiple authors but are actually part of titles
    author = author.replace('et al', '')
    author = author.replace('and others', '')
    author = author.replace('et. al', '')
    author = author.replace('et. al.', '')
    
    # Split by common separators
    separators = ['&', 'and', ',', ';', '/']
    for sep in separators:
        if sep in author:
            parts = [p.strip() for p in author.split(sep)]
            # Filter out empty strings and common phrases
            parts = [p for p in parts if p and not p in ['', ' ', 'and', '&', ',', ';', '/']]
            if len(parts) > 1:
                return True
    return False

def extract_metadata(pdf_path):
    """Extract metadata from a PDF file."""
    try:
        # First try with pdfinfo
        result = subprocess.run(['pdfinfo', pdf_path], capture_output=True, text=True)
        if result.returncode == 0:
            info = {}
            for line in result.stdout.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    info[key.strip()] = value.strip()
            
            # Handle page count that might be a float
            if 'Pages' in info:
                try:
                    page_count = int(float(info['Pages']))
                except (ValueError, TypeError):
                    page_count = 0
            else:
                page_count = 0
                
            # Extract author from filename if not in metadata
            filename = os.path.basename(pdf_path)
            author = info.get('Author', '')
            if not author or author.lower() in ['anonymous', 'unknown']:
                if ' - ' in filename:
                    author = filename.split(' - ')[0].strip()
                
            return {
                'title': info.get('Title', ''),
                'author': author,
                'subject': info.get('Subject', ''),
                'keywords': info.get('Keywords', ''),
                'creator': info.get('Creator', ''),
                'producer': info.get('Producer', ''),
                'creation_date': info.get('CreationDate', ''),
                'mod_date': info.get('ModDate', ''),
                'page_count': page_count,
                'file_size': os.path.getsize(pdf_path),
                'file_path': str(pdf_path),
                'file_name': os.path.basename(pdf_path),
                'processed': True
            }
    except Exception as e:
        print(f"Error getting metadata with pdfinfo: {str(e)}")
    
    # Fallback to PyPDF2 if pdfinfo fails
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            info = reader.metadata if reader.metadata else {}
            
            # Handle page count that might be a float
            try:
                page_count = int(float(reader.get('/Pages').get('/Count', 0)))
            except (ValueError, TypeError, AttributeError):
                page_count = 0
                
            # Extract author from filename if not in metadata
            filename = os.path.basename(pdf_path)
            author = info.get('/Author', '')
            if not author or author.lower() in ['anonymous', 'unknown']:
                if ' - ' in filename:
                    author = filename.split(' - ')[0].strip()
                
            return {
                'title': info.get('/Title', ''),
                'author': author,
                'subject': info.get('/Subject', ''),
                'keywords': info.get('/Keywords', ''),
                'creator': info.get('/Creator', ''),
                'producer': info.get('/Producer', ''),
                'creation_date': info.get('/CreationDate', ''),
                'mod_date': info.get('/ModDate', ''),
                'page_count': page_count,
                'file_size': os.path.getsize(pdf_path),
                'file_path': str(pdf_path),
                'file_name': os.path.basename(pdf_path),
                'processed': True
            }
    except Exception as e:
        print(f"Error getting metadata with PyPDF2: {str(e)}")
        return None

def get_or_create_author(author_name):
    """Get or create an author record."""
    if not author_name:
        return 1  # Return ID for "Unknown" author
        
    # Clean the author name
    author_name = author_name.strip()
    if not author_name or author_name.lower() in ['anonymous', 'unknown']:
        return 1
        
    # Try to find the author
    conn = psycopg2.connect(
        dbname="musartao",
        user="datasundae",
        password="6AV%b9",
        host="localhost"
    )
    cur = conn.cursor()
    
    try:
        # First try exact match
        cur.execute("SELECT id FROM authors WHERE name = %s", (author_name,))
        result = cur.fetchone()
        if result:
            return result[0]
            
        # Try case-insensitive match
        cur.execute("SELECT id FROM authors WHERE LOWER(name) = LOWER(%s)", (author_name,))
        result = cur.fetchone()
        if result:
            return result[0]
            
        # Try partial match
        cur.execute("SELECT id FROM authors WHERE %s LIKE '%%' || name || '%%'", (author_name,))
        result = cur.fetchone()
        if result:
            return result[0]
            
        # Create new author if not found
        cur.execute("INSERT INTO authors (name) VALUES (%s) RETURNING id", (author_name,))
        author_id = cur.fetchone()[0]
        conn.commit()
        return author_id
        
    except Exception as e:
        print(f"Error in get_or_create_author: {str(e)}")
        return 1  # Return ID for "Unknown" author
    finally:
        cur.close()
        conn.close()

def insert_metadata(metadata):
    """Insert book metadata into the database."""
    try:
        conn = psycopg2.connect(
            dbname="musartao",
            user="datasundae",
            password="6AV%b9",
            host="localhost"
        )
        cur = conn.cursor()
        
        # Get or create author
        author_id = None
        if metadata['author']:
            author_id = get_or_create_author(metadata['author'])
            print(f"Got author_id: {author_id}")
        
        # Parse creation date if available
        created_at = None
        if metadata.get('creation_date'):
            try:
                # Try to parse D:YYYYMMDDHHmmSS format
                date_str = metadata['creation_date']
                if date_str.startswith('D:'):
                    date_str = date_str[2:]
                created_at = datetime.strptime(date_str[:14], '%Y%m%d%H%M%S')
            except (ValueError, TypeError):
                try:
                    # Try ISO format
                    created_at = datetime.fromisoformat(metadata['creation_date'])
                except (ValueError, TypeError):
                    created_at = None
        
        # Insert book metadata
        cur.execute("""
            INSERT INTO books (
                filename,
                file_path,
                processed,
                created_at,
                page_count,
                title,
                author,
                publication_date,
                publisher,
                language,
                description,
                thumbnail,
                lifespan,
                author_id
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
        """, (
            metadata['file_name'],
            metadata['file_path'],
            True,
            created_at,
            metadata.get('page_count', 0),
            metadata.get('title', ''),
            metadata.get('author', ''),
            None,  # publication_date
            metadata.get('producer', ''),
            None,  # language
            metadata.get('subject', ''),
            None,  # thumbnail
            None,  # lifespan
            author_id
        ))
        
        conn.commit()
        print(f"Successfully inserted metadata for {metadata['file_name']}")
        
    except Exception as e:
        print(f"Error inserting metadata: {str(e)}")
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

def update_author_lifespan(author_name, birth_year, death_year):
    """Update author's lifespan information."""
    conn = psycopg2.connect(
        dbname="musartao",
        user="datasundae",
        password="6AV%b9",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()
    
    try:
        cur.execute("""
            UPDATE authors 
            SET birth_year = %s, death_year = %s
            WHERE name = %s
        """, (birth_year, death_year, author_name))
        conn.commit()
        print(f"Updated lifespan for author: {author_name}")
    except Exception as e:
        print(f"Error updating author lifespan: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def merge_author_records(primary_name, alternate_names):
    """Merge multiple author records into one, updating all book references."""
    conn = psycopg2.connect(
        dbname="musartao",
        user="datasundae",
        password="6AV%b9",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()
    
    try:
        # Get or create the primary author record
        cur.execute("SELECT id FROM authors WHERE name = %s", (primary_name,))
        result = cur.fetchone()
        if result:
            primary_id = result[0]
        else:
            cur.execute("INSERT INTO authors (name) VALUES (%s) RETURNING id", (primary_name,))
            primary_id = cur.fetchone()[0]
        
        # Update books referencing alternate names to use primary_id
        for alt_name in alternate_names:
            cur.execute("SELECT id FROM authors WHERE name = %s", (alt_name,))
            alt_result = cur.fetchone()
            if alt_result:
                alt_id = alt_result[0]
                # Update book references
                cur.execute("UPDATE books SET author_id = %s WHERE author_id = %s", (primary_id, alt_id))
                # Delete the alternate author record
                cur.execute("DELETE FROM authors WHERE id = %s", (alt_id,))
        
        conn.commit()
        print(f"Successfully merged author records into {primary_name}")
    except Exception as e:
        print(f"Error merging author records: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def normalize_author_names():
    """Normalize author names by merging duplicate records."""
    # Define groups of equivalent author names
    author_groups = [
        {
            "primary": "Friedrich Wilhelm Nietzsche",
            "alternates": [
                "Friedrich Nietzsche",
                "Nietzsche, Friedrich Wilhelm",
                "Nietzsche, Friedrich"
            ]
        },
        {
            "primary": "Jean Piaget",
            "alternates": [
                "JEAN PIAGET",
                "Jean PIaget",
                "Piaget"
            ]
        },
        {
            "primary": "Carl Gustav Jung",
            "alternates": [
                "Jung, C. G.",
                "Jung, Carl",
                "Jung, Carl Gustav"
            ]
        },
        {
            "primary": "Max Weber",
            "alternates": [
                "Weber, Max"
            ]
        },
        {
            "primary": "Martin Heidegger",
            "alternates": [
                "Heidegger, Martin"
            ]
        },
        {
            "primary": "Allen B. Downey",
            "alternates": [
                "Downey, Allen B."
            ]
        },
        {
            "primary": "Brian Tarquin",
            "alternates": [
                "Tarquin, Brian"
            ]
        },
        {
            "primary": "Mark Levine",
            "alternates": [
                "Levine, Mark"
            ]
        },
        {
            "primary": "Matt Picone",
            "alternates": [
                "Picone, Matt"
            ]
        },
        {
            "primary": "Lao Tsuo",
            "alternates": [
                "Tsuo, Lao"
            ]
        }
    ]
    
    # Merge each group
    for group in author_groups:
        merge_author_records(group["primary"], group["alternates"])

def main():
    if len(sys.argv) > 1:
        # Use the provided PDF path
        pdf_path = Path(sys.argv[1])
    else:
        # Default to processing a single book from PDFIngest directory
        pdf_dir = Path("/Volumes/NVME_Expansion/musartao/data/books/PDFIngest")
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            print("No PDF files found in the PDFIngest directory")
            return
        
        # Process the first PDF file
        pdf_path = pdf_files[0]
    
    print(f"Processing: {pdf_path}")
    
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        return
    
    metadata = extract_metadata(pdf_path)
    insert_metadata(metadata)

if __name__ == "__main__":
    main() 