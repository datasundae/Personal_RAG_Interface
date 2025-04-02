import os
import psycopg2
from pathlib import Path
import subprocess
from datetime import datetime
from process_book_metadata import extract_metadata, insert_metadata, has_multiple_authors

def delete_all_records():
    """Delete all records from the books table."""
    conn = psycopg2.connect(
        dbname="musartao",
        user="datasundae",
        password="6AV%b9",
        host="localhost",
        port="5432"
    )
    
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM books")
            conn.commit()
            print("Successfully deleted all records from the books table")
    except Exception as e:
        print(f"Error deleting records: {str(e)}")
        conn.rollback()
    finally:
        conn.close()

def process_all_books():
    """Process all PDF files in the specified directory."""
    # Set the directory for processed PDFs
    pdf_dir = Path("/Volumes/NVME_Expansion/musartao/data/books/Processed")
    
    # Get all PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF file
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        
        try:
            # Extract metadata
            metadata = extract_metadata(str(pdf_file))
            if not metadata:
                print(f"Failed to extract metadata for {pdf_file.name}")
                continue
                
            # Check for multiple authors
            if has_multiple_authors(metadata['author']):
                print(f"Skipping {pdf_file.name} - multiple authors detected")
                continue
                
            # Insert metadata into database
            print(f"Attempting to insert metadata for {pdf_file.name}")
            print(f"Author: {metadata['author']}")
            
            insert_metadata(metadata)
            print(f"Successfully processed: {pdf_file.name}")
            
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {str(e)}")
            continue
    
    print("\nBook processing completed!")

def main():
    print("Starting book processing...")
    
    # Delete existing records
    print("\nDeleting existing records...")
    delete_all_records()
    
    # Process all books
    print("\nProcessing all books...")
    process_all_books()
    
    print("\nBook processing completed!")

if __name__ == "__main__":
    main() 