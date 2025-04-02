from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from dotenv import load_dotenv
from src.postgres_vector_db import PostgreSQLVectorDB
from src.rag_document import RAGDocument
from openai import OpenAI
import tiktoken
import logging
import warnings
from functools import wraps
import secrets
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from google.auth.transport import requests
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import json
from datetime import timedelta, datetime
import psycopg2
import re
from flask_session import Session
from typing import List

# Load environment variables
load_dotenv()

# Suppress specific warnings about tokenizers
warnings.filterwarnings("ignore", message=".*tokenizers.*")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Allow OAuthlib to not use HTTPS for local development
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(32))
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
app.config['SESSION_COOKIE_SECURE'] = False  # Set to False for development
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = None  # Allow cross-site requests for OAuth
app.config['SESSION_TYPE'] = 'filesystem'  # Use filesystem for session storage
app.config['SESSION_FILE_DIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flask_session')
app.config['SESSION_FILE_THRESHOLD'] = 100
app.config['SESSION_FILE_MODE'] = 0o600
app.config['SESSION_USE_SIGNER'] = True  # Sign the session cookie
app.config['SESSION_KEY_PREFIX'] = 'rag_ai:'  # Prefix for session keys
app.config['SESSION_REFRESH_EACH_REQUEST'] = True  # Refresh session on each request
app.config['SESSION_COOKIE_NAME'] = 'rag_ai_session'  # Custom session cookie name
app.config['SESSION_COOKIE_PATH'] = '/'  # Ensure cookie is available for all paths
app.config['SESSION_COOKIE_DOMAIN'] = None  # Allow cookie to work on all domains in development
app.config['SESSION_COOKIE_EXPIRES'] = timedelta(minutes=30)  # Set cookie expiration
app.config['SESSION_COOKIE_MAX_AGE'] = 1800  # 30 minutes in seconds

# Initialize Flask-Session
Session(app)

# Ensure session directory exists
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize vector database
vector_db = PostgreSQLVectorDB(connection_string='postgresql://datasundae:6AV%25b9@localhost:5432/musartao')

# Initialize sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Google OAuth2 configuration
CLIENT_SECRETS_FILE = 'google_client_secret_804506683754-9ogj9ju96r0e88fb6v7t7usga753hh0h.apps.googleusercontent.com.json'

# Load client secrets
with open(CLIENT_SECRETS_FILE) as f:
    client_secrets = json.load(f)
    GOOGLE_CLIENT_ID = client_secrets['web']['client_id']
    GOOGLE_CLIENT_SECRET = client_secrets['web']['client_secret']

GOOGLE_REDIRECT_URI = os.getenv('GOOGLE_REDIRECT_URI', 'http://127.0.0.1:5009/callback')
ALLOWED_DOMAINS = ['datasundae.com']  # List of allowed email domains

# OAuth2 scopes
SCOPES = [
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile',
    'openid'
]

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            logger.warning(f"User not authenticated. Session data: {session}")
            # For API requests, return 401 instead of redirecting
            if request.is_json:
                return jsonify({'error': 'Authentication required'}), 401
            # For web requests, redirect to login
            return redirect(url_for('login'))
        logger.info(f"User authenticated: {session.get('user_id')}")
        return f(*args, **kwargs)
    return decorated_function

@app.before_request
def before_request():
    """Initialize session before each request."""
    if not session.get('_fresh'):
        session['_fresh'] = True
        session.permanent = True
        logger.info(f"Session initialized: {session}")
    # Ensure session is marked as modified to trigger save
    session.modified = True

@app.after_request
def after_request(response):
    """Ensure session is saved after each request."""
    session.modified = True
    return response

def init_models():
    """Initialize models after the fork to avoid tokenizer warnings"""
    global vector_db, model
    if vector_db is None:
        try:
            vector_db = PostgreSQLVectorDB()
            logger.info("Successfully connected to vector database")
        except Exception as e:
            logger.error(f"Failed to connect to vector database: {str(e)}")
            raise

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-4-turbo-preview")
    return len(encoding.encode(text))

def get_relevant_context(query: str) -> List[str]:
    """Get relevant context from the vector database."""
    try:
        logger.info(f"Attempting to retrieve context from books database...")
        logger.info(f"Searching for context related to: {query}")
        
        # Generate query embedding
        logger.info("Generating query embedding...")
        query_embedding = model.encode(query)
        logger.info(f"Generated embedding with shape: {query_embedding.shape}")
        
        # Perform vector similarity search with metadata filter for Animal Spirits
        logger.info("Performing vector similarity search...")
        metadata_filter = {
            "title": "Animal Spirits",
            "author": "Robert Schiller"
        }
        logger.info(f"Using metadata filter: {metadata_filter}")
        results = vector_db.search(query_embedding, limit=5, metadata_filter=metadata_filter)
        
        if not results:
            logger.info("No relevant documents found")
            return []
            
        logger.info(f"Found {len(results)} relevant documents")
        
        # Process and return context
        context_parts = []
        for i, doc in enumerate(results, 1):
            logger.info(f"Processing document {i}/{len(results)}")
            logger.info(f"Document {i} metadata: {doc.metadata}")  # Log full metadata for debugging
            
            if not doc.text:
                logger.warning(f"Document {i} has no text content")
                continue
            
            # Format the document with its metadata
            formatted_doc = f"""
From '{doc.metadata.get('title', 'Unknown Title')}' by {doc.metadata.get('author', 'Unknown Author')}
Page: {doc.metadata.get('page', 'Unknown')}
Similarity Score: {doc.metadata.get('similarity', 'Unknown')}

Content:
{doc.text}
"""
            context_parts.append(formatted_doc)
            
        return context_parts
        
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        return []

def filter_sensitive_info(text):
    """Filter out sensitive information from text."""
    # Add patterns for sensitive information
    patterns = [
        r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',  # SSN
        r'\b\d{16}\b',  # Credit card numbers
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
        r'\b\d{10}\b',  # Phone numbers
        r'\b\d{9}\b',  # Account numbers
    ]
    
    filtered_text = text
    for pattern in patterns:
        filtered_text = re.sub(pattern, '[REDACTED]', filtered_text)
    return filtered_text

@app.route('/login', methods=['GET', 'POST'])
def login():
    # If user is already logged in, redirect to home
    if 'user_id' in session:
        logger.info(f"User already logged in: {session.get('user_id')}")
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        return redirect(url_for('google_login'))
        
    # Clear any existing session data
    session.clear()
    logger.info("Session cleared for new login")
    return render_template('login.html')

@app.route('/google_login')
def google_login():
    """Redirect to Google OAuth2 login."""
    try:
        flow = Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE,
            scopes=SCOPES,
            redirect_uri=GOOGLE_REDIRECT_URI
        )
        
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true'
        )
        
        session['state'] = state
        logger.info(f"Stored state in session: {state}")
        logger.info(f"Redirecting to Google OAuth2 login with URL: {authorization_url}")
        return redirect(authorization_url)
        
    except Exception as e:
        logger.error(f"Error in google_login: {str(e)}")
        session.clear()
        return redirect(url_for('login'))

@app.route('/callback')
def callback():
    """Handle Google OAuth2 callback."""
    try:
        if 'state' not in session:
            logger.error("No state found in session")
            return redirect(url_for('login'))
            
        state = session['state']
        logger.info(f"Retrieved state from session: {state}")
        logger.info(f"Callback URL: {request.url}")
        
        flow = Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE,
            scopes=SCOPES,
            state=state,
            redirect_uri=GOOGLE_REDIRECT_URI
        )
        
        logger.info("Fetching token...")
        flow.fetch_token(
            authorization_response=request.url
        )
        logger.info("Token fetched successfully")
        
        credentials = flow.credentials
        logger.info("Verifying ID token...")
        id_info = id_token.verify_oauth2_token(
            credentials.id_token, requests.Request(), GOOGLE_CLIENT_ID
        )
        logger.info("ID token verified successfully")
        
        # Check if email domain is allowed
        email = id_info['email']
        domain = email.split('@')[1]
        
        if domain not in ALLOWED_DOMAINS:
            logger.error(f"Access denied for domain: {domain}")
            return "Access denied. Only datasundae.com email addresses are allowed.", 403
        
        # Store user info in session
        session['user_id'] = email
        session['user_name'] = id_info.get('name', email)
        session['credentials'] = {
            'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes
        }
        
        # Make session permanent and ensure it's saved
        session.permanent = True
        session.modified = True
        
        logger.info(f"Successfully logged in user: {email}")
        logger.info(f"Session data after login: {session}")
        return redirect(url_for('home'))
        
    except Exception as e:
        logger.error(f"Error in callback: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")
        session.clear()
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    """Clear the session and log out the user."""
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def home():
    return render_template('home.html')

@app.route('/ingest', methods=['POST'])
@login_required
def ingest_content():
    """Ingest content into the vector database."""
    try:
        content = request.json.get('content')
        person = request.json.get('person')  # New field for person tagging
        
        if not content:
            return jsonify({'error': 'No content provided'}), 400
            
        # Initialize models if needed
        init_models()
        
        # Create metadata with person tag if provided
        metadata = {
            'source': 'plain_text',
            'ingested_by': session.get('user_id'),
            'ingestion_date': datetime.now().isoformat()
        }
        if person:
            metadata['person'] = person
        
        # Create RAG document
        doc = RAGDocument(text=content, metadata=metadata)
        
        # Add to vector database
        doc_ids = vector_db.add_documents([doc])
        
        logger.info("Successfully ingested plain text content")
        return jsonify({
            'message': 'Content successfully ingested',
            'doc_ids': doc_ids,
            'metadata': metadata
        }), 200
        
    except Exception as e:
        logger.error(f"Error ingesting content: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    try:
        message = request.json.get('message')
        if not message:
            return jsonify({'error': 'No message provided'}), 400

        logger.info(f"Received user message: {message}")
        
        # Get relevant context from books database
        logger.info("Attempting to retrieve context from books database...")
        context = get_relevant_context(message)
        
        # Even if no context is found, we should still proceed with a response
        if not context:
            logger.info("No relevant context found - proceeding with general knowledge")
            context = "No specific information found in the database for this query."
        
        # Prepare the system message with context
        system_message = f"""You are an AI assistant with access to information from a collection of books. 
        Here are the most relevant passages from the books for the user's question:

        {'\n\n'.join(context) if context else "No specific information found in the database for this query."}
        
        RESPONSE GUIDELINES:
        1. Use the provided book passages to answer questions, citing the specific books and authors
        2. If the information is not directly available in the passages, say so clearly
        3. If multiple books are relevant, mention them all and explain how they relate to the question
        4. If no relevant information is found in the passages, say so clearly and try to provide a helpful general response
        5. For non-English content, consider the original language and provide translations when possible
        
        If the passages are not relevant, respond based on your general knowledge while being clear about the source of information."""

        logger.info("Sending request to OpenAI with context")
        
        # Get response from OpenAI
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": message}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        logger.info("Received response from OpenAI")
        return jsonify({
            'response': response.choices[0].message.content,
            'has_context': bool(context and context != "No specific information found in the database for this query.")
        }), 200

    except Exception as e:
        error_message = str(e)
        logger.error(f"Error in chat endpoint: {error_message}")
        if "rate_limit_exceeded" in error_message:
            return jsonify({
                'error': 'The request was too large. Please try again with a shorter message or wait a moment.'
            }), 429
        return jsonify({
            'error': 'An error occurred while processing your request. Please try again.',
            'details': error_message if app.debug else None
        }), 500

def truncate_text(text: str, max_tokens: int) -> str:
    """Truncate text to fit within token limit.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens allowed
        
    Returns:
        Truncated text
    """
    # Simple implementation - split by words and take first N words
    # This is a rough approximation since we're not using the actual tokenizer
    words = text.split()
    return " ".join(words[:max_tokens]) + "..."

if __name__ == '__main__':
    app.run(debug=True, port=5009) 