import psycopg2
from psycopg2.extras import DictCursor

def check_table_structure():
    """Check the structure of the existing books table."""
    db_params = {
        'dbname': 'musartao',
        'user': 'datasundae',
        'password': '6AV%b9',
        'host': 'localhost'
    }
    
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor(cursor_factory=DictCursor)
        
        # Get table structure
        cur.execute("""
            SELECT column_name, data_type, character_maximum_length, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'books'
            ORDER BY ordinal_position;
        """)
        
        columns = cur.fetchall()
        
        print("Current books table structure:")
        print("-" * 80)
        for col in columns:
            print(f"Column: {col['column_name']}")
            print(f"Type: {col['data_type']}")
            if col['character_maximum_length']:
                print(f"Max length: {col['character_maximum_length']}")
            print(f"Nullable: {col['is_nullable']}")
            print("-" * 80)
            
        # Check indexes
        cur.execute("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'books';
        """)
        
        indexes = cur.fetchall()
        
        print("\nExisting indexes:")
        print("-" * 80)
        for idx in indexes:
            print(f"Index: {idx['indexname']}")
            print(f"Definition: {idx['indexdef']}")
            print("-" * 80)
            
    except Exception as e:
        print(f"Error checking table structure: {str(e)}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    check_table_structure() 