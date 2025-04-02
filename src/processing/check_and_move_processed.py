#!/usr/bin/env python3
import os
import shutil
import psycopg2
from pathlib import Path

def get_processed_files_from_db():
    """Get list of processed files from database."""
    conn = psycopg2.connect(
        dbname="musartao",
        user="datasundae",
        password="6AV%b9",
        host="localhost",
        port=5432
    )
    
    try:
        cur = conn.cursor()
        # Query to get all files from the books table
        cur.execute("SELECT filename FROM books;")
        rows = cur.fetchall()
        return [row[0] for row in rows if row[0]]
    finally:
        cur.close()
        conn.close()

def main():
    # Directories
    books_dir = "/Volumes/NVME_Expansion/musartao/data/books"
    processed_dir = os.path.join(books_dir, "Processed")
    text_files_dir = os.path.join(books_dir, "Text Files")
    
    # Create directories if they don't exist
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(text_files_dir, exist_ok=True)
    
    # Get list of processed files from database
    processed_filenames = get_processed_files_from_db()
    print(f"\nFound {len(processed_filenames)} processed files in database:")
    for filename in sorted(processed_filenames):
        print(f"  - {filename}")
    
    # Get list of files in processed directory
    processed_files = [f for f in os.listdir(processed_dir) if f.endswith('.pdf')]
    print(f"\nFound {len(processed_files)} PDF files in Processed directory:")
    for pdf in sorted(processed_files):
        print(f"  - {pdf}")
    
    # Find files in Processed directory that don't have a database record
    print("\nFiles in Processed directory without database records:")
    missing_from_db = []
    for pdf in processed_files:
        if pdf not in processed_filenames:
            missing_from_db.append(pdf)
            print(f"  - {pdf}")
    
    # Find files in database that aren't in Processed directory
    print("\nFiles in database but missing from Processed directory:")
    missing_from_processed = []
    for filename in processed_filenames:
        if filename.endswith('.pdf') and filename not in processed_files:
            missing_from_processed.append(filename)
            print(f"  - {filename}")
    
    print(f"\nSummary:")
    print(f"- Database records: {len(processed_filenames)}")
    print(f"- Files in Processed directory: {len(processed_files)}")
    print(f"- Files in Processed without DB records: {len(missing_from_db)}")
    print(f"- Files in DB missing from Processed: {len(missing_from_processed)}")
    
    # Move unprocessed files back to main directory
    print("\nMoving unprocessed files back to main directory...")
    for filename in missing_from_db:
        src = os.path.join(processed_dir, filename)
        dst = os.path.join(books_dir, filename)
        if os.path.exists(src):
            print(f"Moving {filename} back to main directory...")
            shutil.move(src, dst)
    
    # Move text files to Text Files directory
    print("\nMoving text files to Text Files directory...")
    for filename in os.listdir(books_dir):
        if filename.endswith('.txt'):
            src = os.path.join(books_dir, filename)
            dst = os.path.join(text_files_dir, filename)
            if os.path.exists(src) and not os.path.exists(dst):
                print(f"Moving {filename} to Text Files directory...")
                shutil.move(src, dst)
    
    # Also check Processed directory for text files
    for filename in os.listdir(processed_dir):
        if filename.endswith('.txt'):
            src = os.path.join(processed_dir, filename)
            dst = os.path.join(text_files_dir, filename)
            if os.path.exists(src) and not os.path.exists(dst):
                print(f"Moving {filename} from Processed to Text Files directory...")
                shutil.move(src, dst)

if __name__ == "__main__":
    main() 