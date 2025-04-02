from typing import Dict, Any, List, Optional
from enum import Enum
import json
from pathlib import Path
from datetime import datetime

class Genre(str, Enum):
    """Document genres."""
    RESEARCH_PAPER = "research_paper"
    ARTICLE = "article"
    BOOK = "book"
    REPORT = "report"
    DOCUMENTATION = "documentation"
    OTHER = "other"

class SubGenre(str, Enum):
    """Document subgenres."""
    MACHINE_LEARNING = "machine_learning"
    ARTIFICIAL_INTELLIGENCE = "artificial_intelligence"
    DATA_SCIENCE = "data_science"
    COMPUTER_SCIENCE = "computer_science"
    SOFTWARE_ENGINEERING = "software_engineering"
    OTHER = "other"

class MetadataSchema:
    """Schema for document metadata validation."""
    
    def __init__(self):
        self.required_fields = {
            "genre": [g.value for g in Genre],
            "subgenre": [sg.value for sg in SubGenre],
            "topics": list,
            "year": int
        }
        self.optional_fields = {
            "author": str,
            "institution": str,
            "keywords": list,
            "language": str,
            "page_count": int,
            "url": str
        }
    
    def validate(self, metadata: Dict[str, Any]) -> bool:
        """Validate metadata against the schema."""
        # Check required fields
        for field, field_type in self.required_fields.items():
            if field not in metadata:
                raise ValueError(f"Missing required field: {field}")
            
            if isinstance(field_type, list):
                if metadata[field] not in field_type:
                    raise ValueError(f"Invalid value for {field}. Must be one of: {field_type}")
            elif not isinstance(metadata[field], field_type):
                raise ValueError(f"Invalid type for {field}. Expected {field_type}, got {type(metadata[field])}")
        
        # Check optional fields if present
        for field, field_type in self.optional_fields.items():
            if field in metadata and not isinstance(metadata[field], field_type):
                raise ValueError(f"Invalid type for {field}. Expected {field_type}, got {type(metadata[field])}")
        
        return True

class MetadataManager:
    """Manage document metadata templates and validation."""
    
    def __init__(self, template_file: Optional[str] = None):
        """Initialize the metadata manager.
        
        Args:
            template_file: Optional path to JSON file containing metadata templates
        """
        self.templates = {}
        if template_file:
            try:
                with open(template_file, 'r') as f:
                    self.templates = json.load(f)
            except:
                print(f"Warning: Could not load template file {template_file}")
                self._init_default_templates()
        else:
            self._init_default_templates()
    
    def _init_default_templates(self):
        """Initialize default metadata templates."""
        self.templates = {
            "research_paper": {
                "genre": Genre.RESEARCH_PAPER.value,
                "subgenre": None,
                "topics": [],
                "year": None,
                "authors": [],
                "institution": None,
                "publication": None
            },
            "article": {
                "genre": Genre.ARTICLE.value,
                "subgenre": None,
                "topics": [],
                "year": None,
                "authors": [],
                "publication": None,
                "url": None
            },
            "book": {
                "genre": Genre.BOOK.value,
                "subgenre": None,
                "topics": [],
                "year": None,
                "authors": [],
                "publisher": None,
                "isbn": None
            },
            "report": {
                "genre": Genre.REPORT.value,
                "subgenre": None,
                "topics": [],
                "year": None,
                "authors": [],
                "organization": None
            },
            "documentation": {
                "genre": Genre.DOCUMENTATION.value,
                "subgenre": None,
                "topics": [],
                "year": None,
                "version": None,
                "product": None
            }
        }
    
    def get_template(self, template_name: str) -> Dict[str, Any]:
        """Get a metadata template by name.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Template dictionary
            
        Raises:
            ValueError if template not found
        """
        template = self.templates.get(template_name.lower())
        if template is None:
            raise ValueError(f"Template not found: {template_name}")
        return template.copy()
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Validate metadata against schema.
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        # Check required fields
        if "genre" not in metadata:
            raise ValueError("Missing required field: genre")
        
        # Validate genre
        if metadata["genre"] not in [g.value for g in Genre]:
            raise ValueError(f"Invalid genre: {metadata['genre']}")
        
        # Validate subgenre if present
        if metadata.get("subgenre") and metadata["subgenre"] not in [sg.value for sg in SubGenre]:
            raise ValueError(f"Invalid subgenre: {metadata['subgenre']}")
        
        # Validate topics
        if "topics" in metadata and not isinstance(metadata["topics"], list):
            raise ValueError("Topics must be a list")
        
        # Validate year if present
        if "year" in metadata and metadata["year"] is not None:
            try:
                year = int(metadata["year"])
                if year < 1900 or year > datetime.now().year:
                    raise ValueError(f"Invalid year: {year}")
            except (TypeError, ValueError):
                raise ValueError(f"Invalid year: {metadata['year']}")
        
        return True
    
    def enrich_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich metadata with additional fields.
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Enriched metadata dictionary
        """
        # Add ingestion timestamp
        metadata["ingestion_date"] = datetime.now().isoformat()
        
        # Add default values for missing fields
        if "topics" not in metadata:
            metadata["topics"] = []
        
        if "genre" not in metadata:
            metadata["genre"] = Genre.OTHER.value
        
        if "subgenre" not in metadata:
            metadata["subgenre"] = SubGenre.OTHER.value
        
        return metadata

# Example usage
if __name__ == "__main__":
    # Initialize metadata manager
    manager = MetadataManager()
    
    # Example metadata
    metadata = {
        "genre": "research_paper",
        "subgenre": "machine_learning",
        "topics": ["AI", "NLP"],
        "year": 2024,
        "author": "John Doe",
        "institution": "University of AI"
    }
    
    # Validate metadata
    try:
        manager.validate_metadata(metadata)
        print("Metadata is valid!")
    except ValueError as e:
        print(f"Invalid metadata: {e}")
    
    # Enrich metadata
    enriched = manager.enrich_metadata(metadata)
    print("\nEnriched metadata:")
    print(json.dumps(enriched, indent=2)) 