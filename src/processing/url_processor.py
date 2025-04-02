import requests
from typing import Dict, Any, Optional
from pathlib import Path
import tempfile
import os
from urllib.parse import urlparse
import re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

from .rag_document import RAGDocument

class URLProcessor:
    """Process documents from URLs."""
    
    def __init__(self):
        """Initialize the URL processor."""
        self.chrome_options = Options()
        self.chrome_options.add_argument('--headless')  # Run in headless mode
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
    
    def process_url(self, url: str, metadata: Optional[Dict[str, Any]] = None) -> RAGDocument:
        """
        Process a document from a URL.
        
        Args:
            url: The URL to process
            metadata: Optional metadata to add to the document
            
        Returns:
            RAGDocument object
        """
        if metadata is None:
            metadata = {}
        
        # Add URL-specific metadata
        url_metadata = {
            "source_url": url,
            "domain": urlparse(url).netloc,
            **metadata
        }
        
        # Process based on URL type
        if "docs.google.com" in url:
            content = self._process_google_doc(url)
        elif any(domain in url for domain in ["arxiv.org", "github.com", "medium.com"]):
            content = self._process_dynamic_page(url)
        else:
            content = self._process_static_page(url)
        
        return RAGDocument(content=content, metadata=url_metadata)
    
    def _process_google_doc(self, url: str) -> str:
        """Process a Google Doc URL."""
        # Extract document ID from URL
        doc_id = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
        if not doc_id:
            raise ValueError("Invalid Google Doc URL")
        
        doc_id = doc_id.group(1)
        export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
        
        # Download the document
        response = requests.get(export_url)
        response.raise_for_status()
        
        return response.text
    
    def _process_dynamic_page(self, url: str) -> str:
        """Process a dynamic webpage using Selenium."""
        driver = webdriver.Chrome(options=self.chrome_options)
        try:
            driver.get(url)
            # Wait for content to load
            time.sleep(5)  # Basic wait, could be improved with explicit waits
            
            # Get the main content
            if "arxiv.org" in url:
                content = driver.find_element(By.CLASS_NAME, "ltx_article").text
            elif "github.com" in url:
                content = driver.find_element(By.CLASS_NAME, "markdown-body").text
            elif "medium.com" in url:
                content = driver.find_element(By.TAG_NAME, "article").text
            else:
                content = driver.find_element(By.TAG_NAME, "body").text
            
            return content
        finally:
            driver.quit()
    
    def _process_static_page(self, url: str) -> str:
        """Process a static webpage."""
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text

# Example usage
if __name__ == "__main__":
    processor = URLProcessor()
    
    # Example Google Doc
    url = "https://docs.google.com/document/d/19s3oIhYcuqaCQxtgL0cFm8G8s-IEJnq561XnF5NxCN0/edit"
    try:
        doc = processor.process_url(url)
        print("Content preview:")
        print("-" * 80)
        print(doc.content[:200] + "...")
        print("-" * 80)
        print("\nMetadata:")
        print(doc.metadata)
    except Exception as e:
        print(f"Error processing URL: {e}") 