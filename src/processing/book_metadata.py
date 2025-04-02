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
import logging

logger = logging.getLogger(__name__)

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
        logger.error(f"Error running pdfinfo: {str(e)}")
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
        logger.error(f"Error with pdftotext: {e}")

    # Try pdfplumber if pdftotext fails
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages_to_process = pdf.pages[:max_pages] if max_pages else pdf.pages[:4]
            for page in pages_to_process:
                try:
                    text += page.extract_text() + "\n"
                except Exception as e:
                    logger.error(f"Error extracting text from page with pdfplumber: {e}")
        if text.strip():
            return text
    except Exception as e:
        logger.error(f"Error with pdfplumber: {e}")

    # Try PyPDF2 as last resort
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = min(len(reader.pages), max_pages if max_pages else 4)
            for page_num in range(num_pages):
                try:
                    text += reader.pages[page_num].extract_text() + "\n"
                except Exception as e:
                    logger.error(f"Error extracting text from page with PyPDF2: {e}")
        if text.strip():
            return text
    except Exception as e:
        logger.error(f"Error with PyPDF2: {e}")

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
        logger.error(f"Error getting metadata with pdfinfo: {str(e)}")
    
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
        logger.error(f"Error getting metadata with PyPDF2: {str(e)}")
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
        logger.error(f"Error in get_or_create_author: {str(e)}")
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
        
        # Insert or update book record
        cur.execute("""
            INSERT INTO books (
                title, author_id, subject, keywords, creator, producer,
                creation_date, mod_date, page_count, file_size, file_path,
                file_name, processed
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) ON CONFLICT (file_path) DO UPDATE SET
                title = EXCLUDED.title,
                author_id = EXCLUDED.author_id,
                subject = EXCLUDED.subject,
                keywords = EXCLUDED.keywords,
                creator = EXCLUDED.creator,
                producer = EXCLUDED.producer,
                creation_date = EXCLUDED.creation_date,
                mod_date = EXCLUDED.mod_date,
                page_count = EXCLUDED.page_count,
                file_size = EXCLUDED.file_size,
                file_name = EXCLUDED.file_name,
                processed = EXCLUDED.processed
            RETURNING id
        """, (
            metadata['title'],
            author_id,
            metadata['subject'],
            metadata['keywords'],
            metadata['creator'],
            metadata['producer'],
            metadata['creation_date'],
            metadata['mod_date'],
            metadata['page_count'],
            metadata['file_size'],
            metadata['file_path'],
            metadata['file_name'],
            metadata['processed']
        ))
        
        book_id = cur.fetchone()[0]
        conn.commit()
        return book_id
        
    except Exception as e:
        logger.error(f"Error inserting metadata: {str(e)}")
        return None
    finally:
        cur.close()
        conn.close()

def update_author_lifespan(author_name, birth_year, death_year):
    """Update author's birth and death years."""
    try:
        conn = psycopg2.connect(
            dbname="musartao",
            user="datasundae",
            password="6AV%b9",
            host="localhost"
        )
        cur = conn.cursor()
        
        cur.execute("""
            UPDATE authors 
            SET birth_year = %s, death_year = %s
            WHERE name = %s
        """, (birth_year, death_year, author_name))
        
        conn.commit()
        
    except Exception as e:
        logger.error(f"Error updating author lifespan: {str(e)}")
    finally:
        cur.close()
        conn.close()

def merge_author_records(primary_name, alternate_names):
    """Merge multiple author records into one."""
    try:
        conn = psycopg2.connect(
            dbname="musartao",
            user="datasundae",
            password="6AV%b9",
            host="localhost"
        )
        cur = conn.cursor()
        
        # Get the primary author's ID
        cur.execute("SELECT id FROM authors WHERE name = %s", (primary_name,))
        primary_id = cur.fetchone()
        if not primary_id:
            logger.error(f"Primary author {primary_name} not found")
            return
        primary_id = primary_id[0]
        
        # Get the alternate authors' IDs
        alternate_ids = []
        for name in alternate_names:
            cur.execute("SELECT id FROM authors WHERE name = %s", (name,))
            result = cur.fetchone()
            if result:
                alternate_ids.append(result[0])
        
        # Update all books with alternate author IDs to use the primary ID
        for alt_id in alternate_ids:
            cur.execute("""
                UPDATE books 
                SET author_id = %s 
                WHERE author_id = %s
            """, (primary_id, alt_id))
        
        # Delete the alternate author records
        for alt_id in alternate_ids:
            cur.execute("DELETE FROM authors WHERE id = %s", (alt_id,))
        
        conn.commit()
        
    except Exception as e:
        logger.error(f"Error merging author records: {str(e)}")
    finally:
        cur.close()
        conn.close()

def normalize_author_names():
    """Normalize author names in the database."""
    try:
        conn = psycopg2.connect(
            dbname="musartao",
            user="datasundae",
            password="6AV%b9",
            host="localhost"
        )
        cur = conn.cursor()
        
        # Get all authors
        cur.execute("SELECT id, name FROM authors")
        authors = cur.fetchall()
        
        for author_id, author_name in authors:
            if not author_name:
                continue
                
            # Clean the name
            cleaned_name = author_name.strip()
            
            # Handle multiple authors
            if has_multiple_authors(cleaned_name):
                # Split the name and create separate records
                parts = [p.strip() for p in cleaned_name.split('&')]
                parts = [p for p in parts if p]
                
                if len(parts) > 1:
                    # Create new records for additional authors
                    for part in parts[1:]:
                        cur.execute("""
                            INSERT INTO authors (name)
                            VALUES (%s)
                            ON CONFLICT (name) DO NOTHING
                        """, (part,))
                    
                    # Update the original record with the first author
                    cur.execute("""
                        UPDATE authors 
                        SET name = %s 
                        WHERE id = %s
                    """, (parts[0], author_id))
            
            # Handle initials
            if re.match(r'^[A-Z]\.\s*[A-Z]\.\s*[A-Za-z\s]+$', cleaned_name):
                # Convert initials to full names if possible
                # This would require a lookup table or external API
                pass
            
            # Handle "et al" and similar
            if re.search(r'\bet\s+al\b', cleaned_name, re.IGNORECASE):
                cleaned_name = re.sub(r'\bet\s+al\b', '', cleaned_name, flags=re.IGNORECASE)
                cleaned_name = cleaned_name.strip()
            
            # Update the author name if it was changed
            if cleaned_name != author_name:
                cur.execute("""
                    UPDATE authors 
                    SET name = %s 
                    WHERE id = %s
                """, (cleaned_name, author_id))
        
        conn.commit()
        
    except Exception as e:
        logger.error(f"Error normalizing author names: {str(e)}")
    finally:
        cur.close()
        conn.close()

def process_book(pdf_path):
    """Process a single book and add it to the database."""
    try:
        # Extract metadata
        metadata = extract_metadata(pdf_path)
        if not metadata:
            logger.error(f"Failed to extract metadata from {pdf_path}")
            return None
            
        # Insert metadata into database
        book_id = insert_metadata(metadata)
        if not book_id:
            logger.error(f"Failed to insert metadata for {pdf_path}")
            return None
            
        return book_id
        
    except Exception as e:
        logger.error(f"Error processing book {pdf_path}: {str(e)}")
        return None 