import psycopg2
from psycopg2.extras import RealDictCursor

def check_books():
    db_params = {
        'dbname': 'musartao',
        'user': 'datasundae',
        'password': '6AV%b9',
        'host': 'localhost',
        'port': 5432
    }
    
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get count of books
        cur.execute("SELECT COUNT(*) as count FROM books;")
        count = cur.fetchone()['count']
        print(f"\nTotal number of books in database: {count}")
        
        # Get sample of books
        cur.execute("""
            SELECT filename, title, author, created_at 
            FROM books 
            ORDER BY created_at DESC 
            LIMIT 5;
        """)
        
        print("\nMost recent books:")
        print("-" * 80)
        for book in cur.fetchall():
            print(f"Filename: {book['filename']}")
            print(f"Title: {book['title']}")
            print(f"Author: {book['author']}")
            print(f"Created At: {book['created_at']}")
            print("-" * 80)
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    check_books() 