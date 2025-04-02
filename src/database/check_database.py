import psycopg2
from psycopg2.extras import RealDictCursor

def check_database_contents():
    # Database connection parameters
    db_params = {
        'dbname': 'musartao',
        'user': 'datasundae',
        'password': '6AV%b9',
        'host': 'localhost'
    }

    try:
        # Connect to the database
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Query to get all books
        cur.execute("""
            SELECT id, filename, title, author, publication_date, publisher, 
                   isbn, language, description, thumbnail, created_at
            FROM books
            ORDER BY id;
        """)

        # Fetch and print results
        books = cur.fetchall()
        print("\nDatabase Contents:")
        print("-" * 80)
        for book in books:
            print(f"\nID: {book['id']}")
            print(f"Filename: {book['filename']}")
            print(f"Title: {book['title']}")
            print(f"Author: {book['author']}")
            print(f"Publication Date: {book['publication_date']}")
            print(f"Publisher: {book['publisher']}")
            print(f"ISBN: {book['isbn']}")
            print(f"Language: {book['language']}")
            print(f"Description: {book['description'][:200] + '...' if book['description'] and len(book['description']) > 200 else book['description']}")
            print(f"Thumbnail: {book['thumbnail']}")
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
    check_database_contents() 