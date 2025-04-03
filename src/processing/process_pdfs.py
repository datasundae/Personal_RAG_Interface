#!/usr/bin/env python3
import os
from pathlib import Path
import psycopg2
from src.database.postgres_vector_db import PostgreSQLVectorDB
from src.processing.ingest_documents import ingest_documents

def update_books_table(file_path: str, processed: bool = True):
    """Update the books table in the musartao database."""
    conn_params = {
        "dbname": "musartao",
        "user": "datasundae",
        "password": "6AV%b9",
        "host": "localhost",
        "port": 5432
    }
    
    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                # Update the processed status in the books table
                cur.execute("""
                    UPDATE books 
                    SET processed = %s 
                    WHERE filename = %s
                """, (processed, os.path.basename(file_path)))
                conn.commit()
    except Exception as e:
        print(f"Error updating books table for {file_path}: {str(e)}")

def main():
    # Initialize the database
    db = PostgreSQLVectorDB()  # Use default parameters from class
    
    # Path to PDFIngest directory
    pdf_dir = "/Volumes/NVME_Expansion/musartao/data/books/PDFIngest"
    processed_dir = "/Volumes/NVME_Expansion/musartao/data/books/Processed"
    
    # Create Processed directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)
    
    # Process PDFs
    try:
        # Process all PDFs in the directory
        doc_ids = ingest_documents(pdf_dir, db, batch_size=5)
        print(f"Successfully processed {len(doc_ids)} documents")
        
        # Move processed files to Processed directory
        for file_path in Path(pdf_dir).glob("*.pdf"):
            try:
                # Update books table to mark as processed
                update_books_table(str(file_path), True)
                
                # Move file to Processed directory
                dest_path = os.path.join(processed_dir, file_path.name)
                os.rename(str(file_path), dest_path)
                print(f"Moved {file_path.name} to Processed directory")
            except Exception as e:
                print(f"Error processing {file_path.name}: {str(e)}")
                
    except Exception as e:
        print(f"Error processing PDFs: {str(e)}")

if __name__ == "__main__":
    main() 