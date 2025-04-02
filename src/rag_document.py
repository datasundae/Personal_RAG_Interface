from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class RAGDocument:
    """A document with text content and metadata."""
    text: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary for serialization."""
        return {
            'text': self.text,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAGDocument':
        """Create document from dictionary."""
        return cls(
            text=data['text'],
            metadata=data['metadata']
        ) 