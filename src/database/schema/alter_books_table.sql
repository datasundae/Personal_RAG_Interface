-- Add new columns to the books table if they don't exist
DO $$
BEGIN
    -- Add page_count if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                  WHERE table_name='books' AND column_name='page_count') THEN
        ALTER TABLE books ADD COLUMN page_count INTEGER;
    END IF;

    -- Add file_path if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                  WHERE table_name='books' AND column_name='file_path') THEN
        ALTER TABLE books ADD COLUMN file_path TEXT;
    END IF;

    -- Add file_name if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                  WHERE table_name='books' AND column_name='file_name') THEN
        ALTER TABLE books ADD COLUMN file_name TEXT;
    END IF;

    -- Add thumbnail if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                  WHERE table_name='books' AND column_name='thumbnail') THEN
        ALTER TABLE books ADD COLUMN thumbnail TEXT;
    END IF;
END $$;

-- Create an index on file_path for faster lookups if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_indexes 
                  WHERE tablename='books' AND indexname='idx_books_file_path') THEN
        CREATE INDEX idx_books_file_path ON books(file_path);
    END IF;
END $$;

-- Update existing records with file information
UPDATE books
SET file_name = filename,
    file_path = '/Volumes/NVME_Expansion/musartao/data/books/' || filename
WHERE file_name IS NULL OR file_path IS NULL; 