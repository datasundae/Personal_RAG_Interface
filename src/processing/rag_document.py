from typing import Dict, Optional

class RAGDocument:
    """A document for RAG with text content and metadata."""
    
    def __init__(self, text: str, metadata: Optional[Dict] = None):
        """Initialize a RAG document.
        
        Args:
            text: The text content of the document
            metadata: Optional metadata dictionary
        """
        self.text = text
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict:
        """Convert the document to a dictionary.
        
        Returns:
            Dict containing text and metadata
        """
        return {
            "text": self.text,
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> "RAGDocument":
        """Create a RAGDocument from a dictionary.
        
        Args:
            data: Dictionary containing text and metadata
            
        Returns:
            RAGDocument instance
        """
        return cls(
            text=data["text"],
            metadata=data.get("metadata", {})
        ) 