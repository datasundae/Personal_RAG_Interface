"""
Web interface configuration.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)

# Flask configuration
FLASK_CONFIG = {
    'SECRET_KEY': os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here'),
    'PERMANENT_SESSION_LIFETIME': 1800,  # 30 minutes
    'SESSION_COOKIE_SECURE': False,  # Set to True in production
    'SESSION_COOKIE_HTTPONLY': True,
    'SESSION_COOKIE_SAMESITE': 'Lax',
    'SESSION_TYPE': 'filesystem',
    'SESSION_FILE_DIR': os.path.join(PROJECT_ROOT, 'flask_session'),
    'SESSION_FILE_THRESHOLD': 100,
    'SESSION_FILE_MODE': 0o600,
    'SESSION_USE_SIGNER': True,
    'SESSION_KEY_PREFIX': 'rag_ai:',
    'SESSION_REFRESH_EACH_REQUEST': True,
}

# Upload configuration
UPLOAD_CONFIG = {
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB max file size
    'UPLOAD_FOLDER': os.path.join(PROJECT_ROOT, 'uploads'),
    'ALLOWED_EXTENSIONS': {'pdf', 'txt', 'docx', 'jpg', 'jpeg', 'png'},
}

# Template configuration
TEMPLATE_CONFIG = {
    'TEMPLATES_AUTO_RELOAD': True,
    'TEMPLATES_DIR': os.path.join(PROJECT_ROOT, 'src', 'web', 'templates'),
}

# Security configuration
SECURITY_CONFIG = {
    'WTF_CSRF_ENABLED': True,
    'WTF_CSRF_SECRET_KEY': os.getenv('CSRF_SECRET_KEY', 'your-csrf-secret-key-here'),
    'WTF_CSRF_TIME_LIMIT': 3600,  # 1 hour
} 