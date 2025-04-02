from process_book_metadata import normalize_author_names
import psycopg2

def normalize_author_names():
    """Normalize author names by merging duplicate records."""
    author_groups = [
        {
            'primary': 'Friedrich Wilhelm Nietzsche',
            'alternates': [
                'Friedrich Nietzsche',
                'Nietzsche, Friedrich Wilhelm',
                'Nietzsche, Friedrich',
                'Friederich Nietzsche'
            ]
        },
        {
            'primary': 'Jean Piaget',
            'alternates': [
                'Piaget',
                'JEAN PIAGET',
                'Jean PIaget',
                'Piaget, Jean'
            ]
        },
        {
            'primary': 'Carl Gustav Jung',
            'alternates': [
                'Carl Jung',
                'Jung, Carl Gustav',
                'Jung, Carl G.',
                'C. G. Jung'
            ]
        },
        {
            'primary': 'Max Weber',
            'alternates': [
                'Weber, Max',
                'M. Weber'
            ]
        },
        {
            'primary': 'Martin Heidegger',
            'alternates': [
                'Heidegger, Martin',
                'M. Heidegger'
            ]
        },
        {
            'primary': 'Allen B. Downey',
            'alternates': [
                'Alan Downey',
                'Allen Downey',
                'Downey, Allen B.'
            ]
        },
        {
            'primary': 'Robert B. Ash',
            'alternates': [
                'Robert Ash',
                'Ash, Robert B.',
                'R. B. Ash'
            ]
        },
        {
            'primary': 'Alfred Adler',
            'alternates': [
                'Adler, Alfred',
                'A. Adler'
            ]
        },
        {
            'primary': 'Ole Olesen-Bagneux',
            'alternates': [
                'Ole Olesen Bagneux',
                'Olesen-Bagneux, Ole'
            ]
        },
        {
            'primary': 'Zhamak Dehghani',
            'alternates': [
                'Dehghani, Zhamak'
            ]
        },
        {
            'primary': 'Joseph Mazur',
            'alternates': [
                'Mazur, Joseph'
            ]
        },
        {
            'primary': 'Matt Weisfeld',
            'alternates': [
                'Weisfeld, Matt'
            ]
        }
    ]
    
    # Connect to the database
    conn = psycopg2.connect(
        dbname="musartao",
        user="datasundae",
        password="6AV%b9",
        host="localhost"
    )
    cur = conn.cursor()
    
    try:
        # Process each author group
        for group in author_groups:
            primary = group['primary']
            alternates = group['alternates']
            
            # Get or create the primary author record
            cur.execute("SELECT id FROM authors WHERE name = %s", (primary,))
            result = cur.fetchone()
            if result:
                primary_id = result[0]
            else:
                cur.execute("INSERT INTO authors (name) VALUES (%s) RETURNING id", (primary,))
                primary_id = cur.fetchone()[0]
            
            # Update all books with alternate author names to use the primary author ID
            for alt in alternates:
                cur.execute("SELECT id FROM authors WHERE name = %s", (alt,))
                result = cur.fetchone()
                if result:
                    alt_id = result[0]
                    # Update books to use primary author ID
                    cur.execute("UPDATE books SET author_id = %s WHERE author_id = %s", (primary_id, alt_id))
                    # Delete the alternate author record
                    cur.execute("DELETE FROM authors WHERE id = %s", (alt_id,))
            
            conn.commit()
            print(f"Successfully merged author records into {primary}")
            
    except Exception as e:
        print(f"Error normalizing author names: {str(e)}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    print("Starting author name normalization...")
    normalize_author_names()
    print("Author name normalization completed.") 