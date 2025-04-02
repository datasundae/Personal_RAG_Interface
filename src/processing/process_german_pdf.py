from src.pdf_processor import PDFProcessor
from src.embeddings import create_embeddings_and_save
import json
from pathlib import Path
import os
import traceback

def analyze_chunks(chunks):
    """Analyze the chunks and print statistics."""
    if not chunks:
        print("\nNo chunks were created after filtering. Please check the filter patterns.")
        return 0
        
    total_tokens = sum(chunk.metadata["token_count"] for chunk in chunks)
    avg_tokens = total_tokens / len(chunks)
    
    print(f"\nChunk Analysis:")
    print(f"Total chunks: {len(chunks)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens per chunk: {avg_tokens:.2f}")
    
    # Print first chunk as sample
    print("\nSample chunk:")
    print("=" * 80)
    print(chunks[0].text[:500] + "..." if len(chunks[0].text) > 500 else chunks[0].text)
    print("=" * 80)
    
    return total_tokens

def process_document():
    # Initialize PDF processor with parameters
    processor = PDFProcessor(
        chunk_size=2500,
        chunk_overlap=200,
        min_chunk_size=500,
        max_chunks=10000,
        language="de"
    )
    
    # Define filter patterns for unwanted text
    filter_patterns = [
        r"Max Weber im Kontext.*?Alle Rechte vorbehalten\.",
        r"Viewlit V\.2\.6.*?InfoSoftWare 1999",
        r"^\s*$",  # Empty lines
        r"^Seite:\s*\d+$",  # Page markers
        r"^[IVX]+\.$",  # Roman numeral section markers
        r"^§\s*\d+$",  # Legal section markers
        r"^Kapitel\s+\d+$",  # Chapter markers
        r"^Abschnitt\s+\d+$",  # Section markers
        r"^Unterabschnitt\s+\d+$",  # Subsection markers
        r"^Anmerkungen$",  # Notes section
        r"^Literaturverzeichnis$",  # Bibliography
        r"^Register$",  # Index
        r"^Inhaltsverzeichnis$",  # Table of contents
        r"^Vorwort$",  # Preface
        r"^Einleitung$",  # Introduction
        r"^Widmung$",  # Dedication
        r"^Danksagung$",  # Acknowledgments
        r"^Impressum$",  # Publication info
        r"^Copyright$",  # Copyright info
        r"^\d+$",  # Standalone numbers
        r"^- Kap\.-Nr\. \d+/\d+ -$",  # Chapter numbers
        r"^- Seite: \d+ -$",  # Page numbers
        r"^- Werke auf CD-ROM -$",  # CD-ROM marker
        r"^Titelseite:.*$",  # Title page
        r"^Titelei:.*$",  # Title info
        r"^Dem Andenken.*$",  # Dedication
        r"^Inhalt:.*$",  # Table of contents
        r"^Schluß.*$",  # Conclusion
        r"^Anhang.*$",  # Appendix
        r"^Bearbeitet von.*$",  # Editor info
        r"^Tübingen \d{4}.*$",  # Publication info
        r"^\[J\.C\.B Mohr.*$",  # Publisher info
        r"^III\. Abteilung.*$",  # Division markers
        r"^Zweiter\s+Teil$",  # Part markers
        r"^Dritter\s+Teil$",
        r"^Erster\s+Teil$",
        r"^Erstes\s+Kapitel$",  # Chapter markers
        r"^Zweites\s+Kapitel$",
        r"^Drittes\s+Kapitel$",
        r"^Viertes\s+Kapitel$",
        r"^Fünftes\s+Kapitel$",
        r"^Sechstes\s+Kapitel$",
        r"^Siebentes\s+Kapitel$",
        r"^Achtes\s+Kapitel$",
        r"^Neuntes\s+Kapitel$",
        r"^Zehntes\s+Kapitel$"
    ]
    
    # Process the PDF file
    pdf_path = "/Users/datasundae/Documents/Wirtschaft und Gesellschaft.pdf"
    metadata = {
        "filter_patterns": filter_patterns,
        "source": "Max Weber - Wirtschaft und Gesellschaft (1922)",
        "language": "de"
    }
    
    # Process the document and get chunks
    chunks = processor.process_pdf(pdf_path, metadata=metadata, return_chunks=True)
    
    # Analyze chunks before creating embeddings
    total_tokens = analyze_chunks(chunks)
    if total_tokens == 0:
        print("No valid chunks were created. Please check the text extraction and filtering.")
        return
    
    # Create embeddings and save
    output_dir = "embeddings/Wirtschaft und Gesellschaft"
    os.makedirs(output_dir, exist_ok=True)
    create_embeddings_and_save(chunks, output_dir)

def main():
    # Process German text
    home = str(Path.home())
    pdf_path = os.path.join(home, "Documents/Wirtschaft und Gesellschaft.pdf")
    process_document()

if __name__ == "__main__":
    main() 