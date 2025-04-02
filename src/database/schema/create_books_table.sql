-- Create books table with enhanced metadata fields
CREATE TABLE IF NOT EXISTS books (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) UNIQUE NOT NULL,
    title VARCHAR(500),
    author VARCHAR(255),
    publication_date VARCHAR(4),
    publisher VARCHAR(255),
    isbn VARCHAR(17),
    language VARCHAR(50),
    description TEXT,
    page_count INTEGER,
    processed_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index on filename for faster lookups
CREATE INDEX IF NOT EXISTS idx_books_filename ON books(filename);

-- Create index on isbn for faster lookups
CREATE INDEX IF NOT EXISTS idx_books_isbn ON books(isbn);

-- Create index on language for filtering
CREATE INDEX IF NOT EXISTS idx_books_language ON books(language);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at
CREATE TRIGGER update_books_updated_at
    BEFORE UPDATE ON books
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column(); 