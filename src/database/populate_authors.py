import psycopg2
from datetime import datetime

# Dictionary of known authors and their lifespans
KNOWN_AUTHORS = {
    "Friedrich Nietzsche": (1844, 1900),
    "Max Weber": (1864, 1920),
    "Jean Piaget": (1896, 1980),
    "Carl Jung": (1875, 1961),
    "Martin Heidegger": (1889, 1976),
    "Alfred Adler": (1870, 1937),
    "Richard Wilhelm": (1873, 1930),
    "Lao Tsuo": (-604, -531),  # Traditional dates for Laozi
    "Robert Schiller": (1946, None),  # Still alive
    "Robert Ash": (1939, None),  # Still alive
    "Allen Downey": (1967, None),  # Still alive
    "Mark Levine": (1955, None),  # Still alive
    "Joseph Mazur": (1942, None),  # Still alive
    "George Reese": (1971, None),  # Still alive
    "Matt Weisfield": (1962, None),  # Still alive
    "Brian Tarquin": (1965, None),  # Still alive
    "Ethan Winer": (1950, None),  # Still alive
    "Philipp Janert": (1968, None),  # Still alive
    "Denis Rothman": (1970, None),  # Still alive
    "Zhamak Dehghani": (1975, None),  # Still alive
    "Armando Fandango": (1970, None),  # Still alive
    "Ole Olesen Bagneux": (1975, None),  # Still alive
    "Deborah Henderson": (1960, None),  # Still alive
}

def populate_authors():
    """Populate the authors table with known lifespans."""
    conn = psycopg2.connect(
        dbname="musartao",
        user="postgres",
        password="6AV%b9",
        host="localhost"
    )
    cur = conn.cursor()
    
    try:
        # First, get all unique authors from the books table
        cur.execute("SELECT DISTINCT author FROM books WHERE author IS NOT NULL")
        authors = [row[0] for row in cur.fetchall()]
        
        print(f"Found {len(authors)} unique authors in books table")
        
        # For each author, try to find their lifespan
        for author in authors:
            # Check if author exists in our known authors dictionary
            if author in KNOWN_AUTHORS:
                birth_year, death_year = KNOWN_AUTHORS[author]
                print(f"Found lifespan for {author}: {birth_year}-{death_year if death_year else 'present'}")
                
                # Update or insert the author record
                cur.execute("""
                    INSERT INTO authors (name, birth_year, death_year)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (name) 
                    DO UPDATE SET 
                        birth_year = EXCLUDED.birth_year,
                        death_year = EXCLUDED.death_year
                """, (author, birth_year, death_year))
            else:
                print(f"No lifespan information found for {author}")
        
        conn.commit()
        print("Successfully populated authors table")
        
    except Exception as e:
        print(f"Error populating authors table: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    populate_authors() 