import os
from typing import Dict, Any
import PyPDF2
from datetime import datetime
import logging

class MetadataExtractor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def extract_pdf_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from a PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf = PyPDF2.PdfReader(file)
                info = pdf.metadata
                
                metadata = {
                    'title': info.get('/Title', ''),
                    'author': info.get('/Author', ''),
                    'subject': info.get('/Subject', ''),
                    'keywords': info.get('/Keywords', ''),
                    'creator': info.get('/Creator', ''),
                    'producer': info.get('/Producer', ''),
                    'creation_date': info.get('/CreationDate', ''),
                    'modification_date': info.get('/ModDate', ''),
                    'page_count': len(pdf.pages),
                    'file_size': os.path.getsize(file_path),
                    'filename': os.path.basename(file_path),
                    'file_path': file_path,
                    'extracted_at': datetime.now().isoformat()
                }
                
                self.logger.info(f"Extracted metadata from {file_path}: {metadata}")
                return metadata
                
        except Exception as e:
            self.logger.error(f"Error extracting metadata from {file_path}: {str(e)}")
            return None
            
    def extract_text_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from a text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                # Read first few lines for potential metadata
                lines = [next(file) for _ in range(10)]
                
                metadata = {
                    'filename': os.path.basename(file_path),
                    'file_path': file_path,
                    'file_size': os.path.getsize(file_path),
                    'first_lines': lines,
                    'extracted_at': datetime.now().isoformat()
                }
                
                self.logger.info(f"Extracted metadata from {file_path}: {metadata}")
                return metadata
                
        except Exception as e:
            self.logger.error(f"Error extracting metadata from {file_path}: {str(e)}")
            return None 