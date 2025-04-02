import os
import sys
import logging
from pathlib import Path
from .book_metadata import process_book, normalize_author_names

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_directory(directory_path):
    """Process all PDF files in a directory."""
    directory = Path(directory_path)
    if not directory.exists():
        logger.error(f"Directory {directory_path} does not exist")
        return
        
    pdf_files = list(directory.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {directory_path}")
    
    for pdf_file in pdf_files:
        try:
            logger.info(f"Processing {pdf_file.name}")
            book_id = process_book(str(pdf_file))
            if book_id:
                logger.info(f"Successfully processed {pdf_file.name} (ID: {book_id})")
            else:
                logger.error(f"Failed to process {pdf_file.name}")
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {str(e)}")

def main():
    """Main function to process books."""
    if len(sys.argv) < 2:
        print("Usage: python -m src.processing.process_books <directory_path>")
        sys.exit(1)
        
    directory_path = sys.argv[1]
    process_directory(directory_path)
    
    # Normalize author names after processing all books
    logger.info("Normalizing author names...")
    normalize_author_names()
    logger.info("Author name normalization complete")

if __name__ == "__main__":
    main() 