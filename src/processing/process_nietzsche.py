#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
from src.rag_document import RAGDocument
from src.postgres_vector_db import PostgreSQLVectorDB
from src.embeddings import create_embeddings_and_save
import json
from tqdm import tqdm
import PyPDF2

# Nietzsche's works mapping
NIETZSCHE_WORKS = {
    'VOL-I': 'The Birth of Tragedy and Other Writings',
    'VOL-II': 'Untimely Meditations',
    'VOL-III': 'Human, All Too Human I',
    'VOL-IV': 'Human, All Too Human II',
    'VOL-V': 'The Gay Science',
    'VOL-VI': 'Thus Spoke Zarathustra',
    'VOL-VII': 'Beyond Good and Evil',
    'VOL-VIII': 'On the Genealogy of Morality',
    'VOL-IX': 'The Case of Wagner and Nietzsche Contra Wagner',
    'VOL-X': 'The Anti-Christ, Ecce Homo, Twilight of the Idols',
    'VOL-XI': 'The Dawn of Day',
    'VOL-XII': 'The Joyful Wisdom',
    'VOL-XIII': 'The Will to Power I',
    'VOL-XIV': 'The Will to Power II',
    'VOL-XV': 'The Antichrist',
    'VOL-XVI': 'Beyond Good and Evil',
    'VOL-XVII': 'The Twilight of the Idols',
    'VOL-XVIII': 'The Will to Power'
}

def read_pdf(file_path):
    """Read text content from a PDF file."""
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def process_nietzsche_works(input_dir, output_dir, embeddings_dir):
    """Process Nietzsche's complete works with proper renaming."""
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Initialize database connections
    vector_db = PostgreSQLVectorDB()
    
    # Process each volume
    for vol_num, title in NIETZSCHE_WORKS.items():
        input_file = f"The-complete-works-of-Friedrich-Nietzsche-{vol_num}.pdf"
        input_path = os.path.join(input_dir, input_file)
        
        if not os.path.exists(input_path):
            print(f"Warning: {input_file} not found")
            continue
            
        # Create new filename
        new_filename = f"Friedrich Nietzsche - {title}.pdf"
        output_path = os.path.join(output_dir, new_filename)
        
        print(f"\nProcessing: {new_filename}")
        
        # Copy and rename file
        shutil.copy2(input_path, output_path)
        
        # Process the PDF
        try:
            # Read PDF content
            text = read_pdf(output_path)
            
            # Create RAG document
            rag_doc = RAGDocument(
                text=text,
                metadata={
                    "author": "Friedrich Nietzsche",
                    "title": title,
                    "volume": vol_num,
                    "source": "Complete Works"
                }
            )
            
            # Create embeddings and save to vector store
            create_embeddings_and_save(
                documents=[rag_doc],
                output_dir=os.path.join(embeddings_dir, f"nietzsche_{vol_num}")
            )
            
            print(f"Successfully processed: {new_filename}")
            
        except Exception as e:
            print(f"Error processing {new_filename}: {str(e)}")
            continue

def main():
    # Define directories
    input_dir = "/Volumes/NVME_Expansion/musartao/data/books"
    output_dir = "/Volumes/NVME_Expansion/musartao/data/books/Processed"
    embeddings_dir = "/Volumes/NVME_Expansion/musartao/data/books/Embeddings"
    
    # Process Nietzsche's works
    process_nietzsche_works(input_dir, output_dir, embeddings_dir)

if __name__ == "__main__":
    main() 