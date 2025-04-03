"""
PDF processor module.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union
import PyPDF2
import io
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np

from ..config.config import DOC_CONFIG
from .rag_document import RAGDocument
from dataclasses import dataclass
from transformers import AutoTokenizer
import re
import tiktoken
import fitz  # PyMuPDF
from PIL import Image

@dataclass
class RAGDocument:
    text: str
    metadata: dict
    
    def to_dict(self):
        """Convert document to dictionary for serialization."""
        return {
            'text': self.text,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create document from dictionary."""
        return cls(
            text=data['text'],
            metadata=data['metadata']
        )

class PDFProcessor:
    """Handles the processing and chunking of PDF documents."""
    
    def __init__(self, 
                 chunk_size: int = DOC_CONFIG["chunk_size"],
                 chunk_overlap: int = DOC_CONFIG["chunk_overlap"],
                 min_chunk_size: int = DOC_CONFIG["min_chunk_size"],
                 max_chunks: int = DOC_CONFIG["max_chunks"],
                 model: str = "gpt-3.5-turbo",
                 language: str = "eng"):
        """Initialize the PDF processor.
        
        Args:
            chunk_size: Maximum number of characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum chunk size to process
            max_chunks: Maximum number of chunks per document
            model: The model name to use for token counting
            language: Language code for OCR (e.g., 'eng' for English, 'deu' for German)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunks = max_chunks
        self.encoding = tiktoken.encoding_for_model(model)
        self.language = language
        
        # Common patterns to filter out
        self.filter_patterns = [
            re.compile(r'Max Weber im Kontext.*?Alle Rechte vorbehalten\.', re.DOTALL),
            re.compile(r'Viewlit V\.2\.6.*?InfoSoftWare 1999'),
            re.compile(r'WG\d{3}'),
            re.compile(r'- Kap\.-Nr\. \d+ \d+ - Seite: \d+'),
            re.compile(r'^\s*\[\d+\]\s*$'),  # Page numbers in brackets
            re.compile(r'^\s*Anmerkungen:\s*$'),  # Footnote sections
            re.compile(r'^\s*Titelseite:.*$'),  # Title page headers
            re.compile(r'^\s*Titelei:.*$'),  # Title headers
        ]
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.encoding.encode(text))
    
    def extract_text_from_pdf(self, file_path: str, metadata: Optional[Dict] = None) -> str:
        """Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            metadata: Optional metadata containing extraction parameters
            
        Returns:
            Extracted text as a string
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
            
        try:
            # Try PyMuPDF first
            doc = fitz.open(file_path)
            text_parts = []
            
            # Skip the first few pages (Google Books preamble)
            start_page = 5
            
            for page_num, page in enumerate(doc[start_page:], start=start_page):
                # Extract text with sorting and ligature preservation
                text = page.get_text("text", sort=True, flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)
                
                # Clean the page text
                text = self._clean_page_text(text)
                
                if text.strip():
                    text_parts.append(text)
                    print(f"Extracted {len(text)} characters from page {page_num + 1} using PyMuPDF")
                    
            doc.close()
            
            if text_parts:
                return "\n\n".join(text_parts)
                
            # If PyMuPDF failed, try OCR
            print("PyMuPDF extraction failed, trying OCR...")
            text_parts = []
            
            # Convert PDF to images
            images = convert_from_path(file_path)
            
            # Skip the first few pages (Google Books preamble)
            for i, image in enumerate(images[start_page:], start=start_page):
                # Convert PIL image to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Convert bytes to PIL Image
                img = Image.open(io.BytesIO(img_byte_arr))
                
                # Extract text using pytesseract with language support
                page_text = pytesseract.image_to_string(img, lang=self.language)
                if page_text:
                    page_text = self._clean_page_text(page_text)
                    if page_text.strip():
                        text_parts.append(page_text)
                        print(f"Extracted {len(page_text)} characters from page {i + 1} using OCR")
                        
            if text_parts:
                return "\n\n".join(text_parts)
                    
            print("Warning: No text was extracted from any page using either method")
            return ""
            
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            raise ValueError(f"Error extracting text from PDF: {str(e)}")
            
    def _clean_page_text(self, text: str) -> str:
        """Clean text extracted from a PDF page.
        
        Args:
            text: Raw text from PDF page
            
        Returns:
            Cleaned text
        """
        # Remove page markers
        text = re.sub(r"Seite:\s*\d+", "", text)
        text = re.sub(r"- Seite: \d+ -", "", text)
        text = re.sub(r"- Kap\.-Nr\. \d+/\d+ -", "", text)
        text = re.sub(r"- Werke auf CD-ROM -", "", text)
        
        # Remove software headers and footers
        text = re.sub(r"Max Weber im Kontext.*?Alle Rechte vorbehalten\.", "", text, flags=re.DOTALL)
        text = re.sub(r"Viewlit V\.2\.6.*?InfoSoftWare 1999", "", text, flags=re.DOTALL)
        
        # Remove standalone numbers (likely page numbers)
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        
        # Remove empty lines
        text = re.sub(r"^\s*$\n?", "", text, flags=re.MULTILINE)
        
        # Fix hyphenation at line breaks
        text = re.sub(r"([a-zäöüß])-\s*\n\s*([a-zäöüß])", r"\1\2", text)
        text = re.sub(r"([A-ZÄÖÜ])-\s*\n\s*([a-zäöüß])", r"\1\2", text)
        
        # Fix spacing around punctuation
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        text = re.sub(r"([.,;:!?])\s+", r"\1 ", text)
        
        # Remove section markers
        text = re.sub(r"^[IVX]+\.\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^§\s*\d+\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^Kapitel\s+\d+\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^Abschnitt\s+\d+\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^Unterabschnitt\s+\d+\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^Anmerkungen\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^Literaturverzeichnis\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^Register\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^Inhaltsverzeichnis\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^Vorwort\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^Einleitung\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^Widmung\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^Danksagung\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^Impressum\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^Copyright\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^Titelseite:.*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^Titelei:.*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^Dem Andenken.*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^Inhalt:.*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^Schluß.*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^Anhang.*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^Bearbeitet von.*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^Tübingen \d{4}.*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\[J\.C\.B Mohr.*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^III\. Abteilung.*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^(Zweiter|Dritter|Erster)\s+Teil\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^(Erstes|Zweites|Drittes|Viertes|Fünftes|Sechstes|Siebentes|Achtes|Neuntes|Zehntes)\s+Kapitel\s*$", "", text, flags=re.MULTILINE)
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        
        return text.strip()
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Remove software headers and footers
        text = re.sub(r'Viewlit V\.2\.6.*?InfoSoftWare 1999', '', text)
        text = re.sub(r'Max Weber im Kontext.*?Alle Rechte vorbehalten\.', '', text)
        
        # Remove page markers and numbers
        text = re.sub(r'^Seite:\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^-\s*Seite:\s*\d+\s*-\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^-\s*Kap\.-Nr\.\s*\d+/\d+\s*-\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\[S\.\s*\d+\]', '', text)
        
        # Fix common OCR errors and special characters
        text = text.replace('ii', 'ü')
        text = text.replace('ae', 'ä')
        text = text.replace('oe', 'ö')
        text = text.replace('ss', 'ß')
        text = text.replace('ﬁ', 'fi')
        text = text.replace('ﬂ', 'fl')
        text = text.replace('ﬀ', 'ff')
        text = text.replace('ﬃ', 'ffi')
        text = text.replace('ﬄ', 'ffl')
        
        # Fix spacing around punctuation and special characters
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        text = re.sub(r'([.,;:!?])\s+', r'\1 ', text)
        text = re.sub(r'\s+([()\[\]{}])', r'\1', text)
        text = re.sub(r'([()\[\]{}])\s+', r'\1', text)
        
        # Fix hyphenation and line breaks
        text = re.sub(r'([a-zäöüß])-\s*\n\s*([a-zäöüß])', r'\1\2', text)
        text = re.sub(r'([A-ZÄÖÜ])-\s*\n\s*([a-zäöüß])', r'\1\2', text)
        
        # Remove section markers and headers
        section_markers = [
            r'^[IVX]+\.\s*$',
            r'^§\s*\d+\s*$',
            r'^Kapitel\s+\d+\s*$',
            r'^Abschnitt\s+\d+\s*$',
            r'^Unterabschnitt\s+\d+\s*$',
            r'^Anmerkungen\s*$',
            r'^Literaturverzeichnis\s*$',
            r'^Register\s*$',
            r'^Inhaltsverzeichnis\s*$',
            r'^Vorwort\s*$',
            r'^Einleitung\s*$',
            r'^Widmung\s*$',
            r'^Danksagung\s*$',
            r'^Impressum\s*$',
            r'^Copyright\s*$',
            r'^Titelseite:.*$',
            r'^Titelei:.*$',
            r'^Dem Andenken.*$',
            r'^Inhalt:.*$',
            r'^Schluß.*$',
            r'^Anhang.*$',
            r'^Bearbeitet von.*$',
            r'^Tübingen \d{4}.*$',
            r'^\[J\.C\.B Mohr.*$',
            r'^III\. Abteilung.*$',
            r'^(Zweiter|Dritter|Erster)\s+Teil\s*$',
            r'^(Erstes|Zweites|Drittes|Viertes|Fünftes|Sechstes|Siebentes|Achtes|Neuntes|Zehntes)\s+Kapitel\s*$'
        ]
        
        for pattern in section_markers:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
        
        # Remove empty lines and normalize line breaks
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        text = re.sub(r'^\s*$\n', '', text, flags=re.MULTILINE)
        
        # Final cleanup
        text = text.strip()
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def find_sentence_boundary(self, text: str, position: int, direction: str = 'forward') -> int:
        """Find the nearest sentence boundary, handling German sentence structures."""
        # German sentence endings (including abbreviations)
        sentence_endings = '.!?'
        abbreviations = ['Dr.', 'Prof.', 'Hr.', 'Fr.', 'Nr.', 'St.', 'str.', 'z.B.', 'd.h.', 'u.a.', 'etc.', 'usw.', 'bzw.', 'ca.']
        
        if direction == 'forward':
            # Look for next sentence boundary
            for i in range(position, len(text)):
                if text[i] in sentence_endings:
                    # Check if it's not part of an abbreviation
                    is_abbreviation = False
                    for abbr in abbreviations:
                        if i >= len(abbr) and text[i-len(abbr)+1:i+1] == abbr:
                            is_abbreviation = True
                            break
                    if not is_abbreviation:
                        # Make sure we include any closing quotes or parentheses
                        while i + 1 < len(text) and text[i+1] in '"\')]':
                            i += 1
                        return i + 1
            return len(text)
        else:
            # Look for previous sentence boundary
            for i in range(position-1, -1, -1):
                if text[i] in sentence_endings:
                    # Check if it's not part of an abbreviation
                    is_abbreviation = False
                    for abbr in abbreviations:
                        if i >= len(abbr)-1 and text[i-len(abbr)+1:i+1] == abbr:
                            is_abbreviation = True
                            break
                    if not is_abbreviation:
                        # Include the sentence ending character
                        return i + 1
                elif i == 0:
                    return 0
            return 0
    
    def _extract_text_with_ocr(self, file_path: str) -> str:
        """Extract text from PDF using OCR with language support."""
        try:
            # Convert PDF to images
            images = convert_from_path(file_path)
            
            # Extract text from each image
            text = ""
            for i, image in enumerate(images):
                # Convert PIL image to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Convert bytes to PIL Image
                img = Image.open(io.BytesIO(img_byte_arr))
                
                # Extract text using pytesseract with language support
                page_text = pytesseract.image_to_string(img, lang=self.language)
                text += f"\n{page_text}"
            
            return text.strip()
            
        except Exception as e:
            raise ValueError(f"Error performing OCR on PDF: {str(e)}")
    
    def process_pdf(self, file_path: str, metadata: Optional[Dict] = None, return_chunks: bool = False) -> Union[RAGDocument, List[RAGDocument]]:
        """
        Process a PDF file and return either a single document or list of chunks.
        
        Args:
            file_path: Path to the PDF file
            metadata: Optional metadata to attach to the document
            return_chunks: Whether to return chunks instead of a single document
            
        Returns:
            Either a RAGDocument or list of RAGDocument objects
        """
        # Extract text from PDF
        raw_text = self.extract_text_from_pdf(file_path)
        
        # Clean the text
        cleaned_text = self.clean_text(raw_text)
        
        # Create document metadata
        doc_metadata = {
            "source": file_path,
            "language": "de",
            "char_count": len(cleaned_text),
            "token_count": len(cleaned_text.split())
        }
        if metadata:
            doc_metadata.update(metadata)
        
        # If return_chunks is True, create chunks
        if return_chunks:
            chunks = self.create_chunks(cleaned_text)
            return chunks
        
        # Otherwise return a single document
        return RAGDocument(text=cleaned_text, metadata=doc_metadata)

    def create_chunks(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Create overlapping chunks of text with token-based sizing."""
        # Tokenize the text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            # Get chunk of tokens
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            
            # Convert back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move start position, accounting for overlap
            start = end - overlap
            
        return chunks
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into logical sections based on headers or major breaks."""
        # Common section markers in German academic texts
        section_patterns = [
            r'\n(?=[IVX]+\.)',  # Roman numerals
            r'\n(?=\d+\.)',     # Arabic numerals
            r'\n(?=[A-Z][A-Za-zäöüßÄÖÜ\s]+(?:\n|$))',  # Capitalized headers
            r'\n(?=§\s*\d+)',   # Legal sections
            r'\n(?=Kapitel\s+\d+)',  # Chapter markers
        ]
        
        # Combine patterns
        pattern = '|'.join(section_patterns)
        
        # Split text into sections
        sections = re.split(pattern, text)
        
        # Clean up sections
        sections = [section.strip() for section in sections if section.strip()]
        
        return sections

    def extract_text(self, pdf_path: str) -> Optional[str]:
        """Extract text from a PDF file using multiple methods."""
        text = self._try_direct_extraction(pdf_path)
        
        # If direct extraction failed or got very little text, try OCR
        if not text or len(text.strip()) < 1000:
            print("Direct extraction failed or got minimal text, trying OCR...")
            text = self._try_ocr_extraction(pdf_path)
            
        return text
    
    def _try_direct_extraction(self, pdf_path: str) -> Optional[str]:
        """Try to extract text directly using PyMuPDF and PyPDF2."""
        try:
            # Try PyMuPDF first
            doc = fitz.open(pdf_path)
            text = ""
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n\n"
                    print(f"Extracted {len(page_text)} characters from page {page_num + 1} using PyMuPDF")
            doc.close()
            
            if text.strip():
                return text
                
            # If PyMuPDF failed, try PyPDF2
            print("PyMuPDF extraction got minimal text, trying PyPDF2...")
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                        print(f"Extracted {len(page_text)} characters from page {page_num + 1} using PyPDF2")
                        
            return text if text.strip() else None
            
        except Exception as e:
            print(f"Direct extraction failed: {str(e)}")
            return None
            
    def _try_ocr_extraction(self, pdf_path: str) -> Optional[str]:
        """Extract text using OCR with Tesseract."""
        try:
            # Convert PDF to images
            print("Converting PDF to images for OCR...")
            images = convert_from_path(pdf_path)
            
            # Process each page
            text = ""
            for i, image in enumerate(images):
                print(f"Processing page {i + 1} with OCR...")
                
                # Convert PIL image to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Convert bytes to PIL Image
                img = Image.open(io.BytesIO(img_byte_arr))
                
                # Extract text using pytesseract
                page_text = pytesseract.image_to_string(img, lang='eng')
                if page_text.strip():
                    text += page_text + "\n\n"
                    print(f"Extracted {len(page_text)} characters from page {i + 1} using OCR")
                
            return text if text.strip() else None
            
        except Exception as e:
            print(f"OCR extraction failed: {str(e)}")
            return None

    def get_page_count(self, file_path: str) -> int:
        """Get the total number of pages in a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Total number of pages
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
            
        try:
            # Try PyMuPDF first
            doc = fitz.open(file_path)
            page_count = len(doc)
            doc.close()
            return page_count
            
        except Exception as e:
            print(f"PyMuPDF error getting page count: {str(e)}")
            try:
                # Try PyPDF2 as fallback
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    return len(reader.pages)
            except Exception as e:
                print(f"PyPDF2 error getting page count: {str(e)}")
                raise ValueError(f"Could not get page count from PDF: {str(e)}") 