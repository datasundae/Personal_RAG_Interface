import psycopg2
from process_book_metadata import has_multiple_authors

def cleanup_multiple_authors():
    """Remove books with multiple authors from the database."""
    conn = psycopg2.connect(
        dbname="musartao",
        user="datasundae",
        password="6AV%b9",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()
    
    try:
        # Get all books
        cur.execute("SELECT id, filename, author FROM books")
        books = cur.fetchall()
        
        # Count books with multiple authors
        multiple_authors_count = 0
        
        # Remove books with multiple authors
        for book_id, filename, author in books:
            if has_multiple_authors(author):
                cur.execute("DELETE FROM books WHERE id = %s", (book_id,))
                multiple_authors_count += 1
                print(f"Removed book with multiple authors: {filename}")
        
        conn.commit()
        print(f"\nTotal books removed: {multiple_authors_count}")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    print("Starting cleanup of books with multiple authors...")
    cleanup_multiple_authors()
    print("Cleanup completed.") 