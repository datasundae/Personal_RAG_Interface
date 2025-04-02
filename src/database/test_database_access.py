import psycopg2
from psycopg2.extras import RealDictCursor
from src.postgres_vector_db import PostgreSQLVectorDB
from src.rag_document import RAGDocument

def test_book_metadata():
    """Test access to book metadata in the musartao database."""
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
        
        # Get all books
        cur.execute("""
            SELECT id, filename, title, author, created_at 
            FROM books 
            ORDER BY created_at DESC;
        """)
        
        books = cur.fetchall()
        print("\nBook Metadata:")
        print("-" * 80)
        for book in books:
            print(f"ID: {book['id']}")
            print(f"Filename: {book['filename']}")
            print(f"Title: {book['title']}")
            print(f"Author: {book['author']}")
            print(f"Created At: {book['created_at']}")
            print("-" * 80)
            
    except Exception as e:
        print(f"Error accessing book metadata: {e}")
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()

def test_vector_search():
    """Test vector search functionality."""
    try:
        # Initialize vector database
        print("\nInitializing vector database...")
        db = PostgreSQLVectorDB()
        
        # Test search
        print("\nTesting vector search...")
        query = "jazz theory and harmony"
        results = db.search(query, k=3)
        
        print(f"\nSearch results for: {query}")
        print("-" * 80)
        for doc, similarity in results:
            print(f"Content: {doc.content[:200]}...")  # Show first 200 chars
            print(f"Metadata: {doc.metadata}")
            print(f"Similarity: {similarity:.3f}")
            print("-" * 80)
            
    except Exception as e:
        print(f"Error testing vector search: {e}")

if __name__ == "__main__":
    print("Testing database access...")
    test_book_metadata()
    test_vector_search() 