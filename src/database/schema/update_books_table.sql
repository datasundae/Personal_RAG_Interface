-- Add processed column if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'books' 
        AND column_name = 'processed'
    ) THEN
        ALTER TABLE books ADD COLUMN processed BOOLEAN DEFAULT FALSE;
    END IF;
END $$;

-- Update processed status for files that were in PDFIngest
UPDATE books 
SET processed = TRUE 
WHERE filename IN (
    'Alfred Adler - The Neurotic Character.pdf',
    'Alfred Adler - Understanding Human Nature.pdf',
    'Alfred Adler - What Life Should Mean to You.pdf',
    'Carl Jung - Aion.pdf',
    'Carl Jung - Archetypes and the Collective Unconscious.pdf',
    'Carl Jung - Man and His Symbols.pdf',
    'Carl Jung - Memories Dreams Reflections.pdf',
    'Carl Jung - Modern Man in Search of a Soul.pdf',
    'Carl Jung - On the Nature of the Psyche.pdf',
    'Carl Jung - Psychological Types.pdf',
    'Carl Jung - Psychology and Religion.pdf',
    'Carl Jung - Psychology of the Unconscious.pdf',
    'Carl Jung - The Archetypes and the Collective Unconscious.pdf',
    'Carl Jung - The Development of Personality.pdf',
    'Carl Jung - The Practice of Psychotherapy.pdf',
    'Carl Jung - The Psychology of the Transference.pdf',
    'Carl Jung - The Red Book.pdf',
    'Carl Jung - The Relations Between the Ego and the Unconscious.pdf',
    'Carl Jung - The Structure and Dynamics of the Psyche.pdf',
    'Carl Jung - The Undiscovered Self.pdf',
    'Carl Jung - Two Essays on Analytical Psychology.pdf',
    'Friedrich Nietzsche - Also Spach Zarasthustra.pdf',
    'Friedrich Nietzsche - Beyond Good and Evil.pdf',
    'Friedrich Nietzsche - The Birth of Tragedy.pdf',
    'Friedrich Nietzsche - The Gay Science.pdf',
    'Friedrich Nietzsche - The Genealogy of Morals.pdf',
    'Friedrich Nietzsche - Thus Spoke Zarathustra.pdf',
    'Friedrich Nietzsche - Twilight of the Idols.pdf'
); 