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
    'SESSION_COOKIE_NAME': 'rag_ai_session',
    'SESSION_COOKIE_PATH': '/',
    'SESSION_COOKIE_DOMAIN': None,
    'SESSION_COOKIE_EXPIRES': 1800,
    'SESSION_COOKIE_MAX_AGE': 1800
}

# Database configuration
DB_CONFIG = {
    'dbname': 'musartao',
    'user': 'datasundae',
    'password': os.getenv('DB_PASSWORD', '6AV%b9'),
    'host': 'localhost',
    'port': '5432'
}

# Google OAuth2 configuration
GOOGLE_CONFIG = {
    'CLIENT_ID': os.getenv('GOOGLE_CLIENT_ID'),
    'CLIENT_SECRET': os.getenv('GOOGLE_CLIENT_SECRET'),
    'REDIRECT_URI': os.getenv('GOOGLE_REDIRECT_URI', 'http://127.0.0.1:5010/callback'),
    'ALLOWED_DOMAINS': ['datasundae.com'],
    'SCOPES': [
        'https://www.googleapis.com/auth/userinfo.email',
        'https://www.googleapis.com/auth/userinfo.profile',
        'openid'
    ]
}

# OpenAI configuration
OPENAI_CONFIG = {
    'API_KEY': os.getenv('OPENAI_API_KEY'),
    'MODEL': 'gpt-4-turbo-preview',
    'MAX_TOKENS': 2000
}

# Vector database configuration
VECTOR_DB_CONFIG = {
    'MODEL_NAME': 'all-MiniLM-L6-v2',
    'MAX_SEARCH_RESULTS': 5,
    'SIMILARITY_THRESHOLD': 0.7
}

# File paths
PATHS = {
    'TEMPLATES': os.path.join(PROJECT_ROOT, 'src', 'web', 'templates'),
    'STATIC': os.path.join(PROJECT_ROOT, 'src', 'web', 'static'),
    'FLASK_SESSION': os.path.join(PROJECT_ROOT, 'flask_session'),
    'DATA': os.path.join(PROJECT_ROOT, 'data'),
    'BOOKS': os.path.join(PROJECT_ROOT, 'data', 'books'),
    'ART': os.path.join(PROJECT_ROOT, 'data', 'art'),
    'FILES': os.path.join(PROJECT_ROOT, 'data', 'files')
} 