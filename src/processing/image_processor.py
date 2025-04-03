"""
Image processor module.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import pytesseract
from PIL import Image
import cv2
import numpy as np

from .rag_document import RAGDocument

class ImageProcessor:
    def __init__(self):
        """Initialize the image processor."""
        pass
    
    def process_image(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> RAGDocument:
        """Process an image file and return a RAGDocument.
        
        Args:
            file_path: Path to the image file
            metadata: Optional metadata to attach to the document
            
        Returns:
            RAGDocument containing the text content and metadata
        """
        try:
            # Read image using OpenCV
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError(f"Could not read image file: {file_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding to preprocess the image
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Convert OpenCV image to PIL Image
            pil_img = Image.fromarray(gray)
            
            # Extract text using pytesseract
            text = pytesseract.image_to_string(pil_img)
            
            if not text.strip():
                raise ValueError(f"Could not extract text from image {file_path}")
            
            # Create metadata if not provided
            if metadata is None:
                metadata = {}
            
            # Add source file information to metadata
            metadata.update({
                "source_file": os.path.basename(file_path),
                "file_type": "image"
            })
            
            return RAGDocument(content=text.strip(), metadata=metadata)
            
        except Exception as e:
            raise ValueError(f"Error processing image {file_path}: {str(e)}")
    
    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results.
        
        Args:
            img: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Apply thresholding
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Apply noise reduction
        gray = cv2.medianBlur(gray, 3)
        
        return gray 