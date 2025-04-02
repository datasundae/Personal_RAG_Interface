"""
Configuration module for application settings.
"""

from .config import DB_CONFIG, DEFAULT_METADATA, DOC_CONFIG
from .metadata_config import MetadataManager, Genre, SubGenre

__all__ = [
    'DB_CONFIG',
    'DEFAULT_METADATA',
    'DOC_CONFIG',
    'MetadataManager',
    'Genre',
    'SubGenre'
] 