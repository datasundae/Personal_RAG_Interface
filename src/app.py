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
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

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
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    logger.error("OPENAI_API_KEY environment variable is not set")
    raise ValueError("OpenAI API key is required")

try:
    client = OpenAI(api_key=api_key)
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    raise

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

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="redis://localhost:6379"
)

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
        
        # Perform vector similarity search without metadata filter
        logger.info("Performing vector similarity search...")
        results = vector_db.search(query, k=5)
        
        if not results:
            logger.info("No relevant documents found")
            return []
            
        logger.info(f"Found {len(results)} relevant documents")
        
        # Process and return context
        context_parts = []
        total_tokens = 0
        max_tokens = 20000  # Limit total tokens to avoid OpenAI rate limits
        max_doc_tokens = 4000  # Maximum tokens per document
        
        for i, (doc, similarity) in enumerate(results, 1):
            logger.info(f"Processing document {i}/{len(results)}")
            logger.info(f"Document {i} metadata: {doc.metadata}")  # Log full metadata for debugging
            
            # Get document text from either text or content attribute
            doc_text = getattr(doc, 'text', None) or getattr(doc, 'content', None)
            if not doc_text:
                logger.warning(f"Document {i} has no text or content")
                logger.warning(f"Document {i} attributes: {dir(doc)}")  # Log available attributes
                continue
            
            logger.info(f"Document {i} text length: {len(doc_text)}")
            
            # Truncate document text if needed
            doc_tokens = count_tokens(doc_text)
            if doc_tokens > max_doc_tokens:
                logger.info(f"Document {i} exceeds token limit ({doc_tokens} tokens), truncating...")
                # Truncate text to roughly max_doc_tokens
                encoding = tiktoken.encoding_for_model("gpt-4-turbo-preview")
                tokens = encoding.encode(doc_text)
                doc_text = encoding.decode(tokens[:max_doc_tokens])
                doc_tokens = count_tokens(doc_text)
                logger.info(f"Document {i} truncated to {doc_tokens} tokens")
            
            # Format the document with its metadata
            formatted_doc = f"""
From '{doc.metadata.get('title', 'Unknown Title')}' by {doc.metadata.get('author', 'Unknown Author')}
Page: {doc.metadata.get('page', 'Unknown')}
Similarity Score: {similarity:.4f}

{doc_text}
"""
            
            # Count tokens in the formatted document
            formatted_tokens = count_tokens(formatted_doc)
            logger.info(f"Document {i} formatted token count: {formatted_tokens}")
            
            # Check if adding this document would exceed the token limit
            if total_tokens + formatted_tokens > max_tokens:
                logger.info(f"Token limit reached ({total_tokens} tokens), stopping context collection")
                break
                
            context_parts.append(formatted_doc)
            total_tokens += formatted_tokens
            
        logger.info(f"Successfully retrieved context from vector database")
        logger.info(f"Context length: {len(context_parts)}")
        logger.info(f"Total tokens: {total_tokens}")
        return context_parts
        
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")
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

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(RateLimitError)
)
def get_openai_response(client, messages, model="gpt-4-turbo-preview", temperature=0.7, max_tokens=1000):
    """Get response from OpenAI with retry logic."""
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    except RateLimitError as e:
        logger.warning(f"OpenAI rate limit hit, will retry: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error in OpenAI API call: {str(e)}")
        raise

@app.route('/chat', methods=['POST'])
@login_required
@limiter.limit("10 per minute")
def chat():
    try:
        message = request.json.get('message')
        if not message:
            return jsonify({'error': 'No message provided'}), 400

        logger.info(f"Received user message: {message}")
        
        # Get relevant context from books database
        logger.info("Attempting to retrieve context from books database...")
        try:
            logger.info("Connecting to vector database...")
            if not vector_db:
                logger.error("Vector database connection is None")
                raise Exception("Vector database connection not initialized")
                
            context = get_relevant_context(message)
            logger.info("Successfully retrieved context from vector database")
            logger.info(f"Context length: {len(context) if context else 0}")
            
        except Exception as e:
            logger.error(f"Error retrieving context from vector database: {str(e)}")
            logger.error(f"Vector DB error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")
            logger.error(f"Vector DB error type: {type(e)}")
            context = []
        
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
        
        try:
            # Verify OpenAI client is initialized
            if not client:
                logger.error("OpenAI client is None")
                raise Exception("OpenAI client not initialized")
                
            # Get response from OpenAI with retry logic
            logger.info("Making OpenAI API call...")
            response = get_openai_response(
                client,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": message}
                ]
            )
            logger.info("Successfully received response from OpenAI")
            
        except RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded after retries: {str(e)}")
            return jsonify({
                'error': 'Rate limit exceeded. Please wait a moment and try again.',
                'retry_after': getattr(e, 'retry_after', 60),  # Default to 60 seconds if not specified
                'source': 'OpenAI'
            }), 429
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            logger.error(f"OpenAI error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")
            logger.error(f"OpenAI error type: {type(e)}")
            return jsonify({
                'error': 'An error occurred while processing your request. Please try again.',
                'details': str(e) if app.debug else None
            }), 500

        return jsonify({
            'response': response.choices[0].message.content,
            'has_context': bool(context and context != "No specific information found in the database for this query.")
        }), 200

    except Exception as e:
        error_message = str(e)
        logger.error(f"Error in chat endpoint: {error_message}")
        logger.error(f"Full error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")
        logger.error(f"Error type: {type(e)}")
        if "maximum context length" in error_message.lower():
            return jsonify({
                'error': 'The request was too large. Please try again with a shorter message.'
            }), 413
        return jsonify({
            'error': 'An error occurred while processing your request. Please try again.',
            'details': error_message if app.debug else None,
            'error_type': str(type(e))
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