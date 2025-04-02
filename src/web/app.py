<<<<<<< HEAD
"""
Web application module.
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from dotenv import load_dotenv
from ..database.postgres_vector_db import PostgreSQLVectorDB
from ..rag_document import RAGDocument
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
from ..processing.book_metadata import process_book, normalize_author_names

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
        
        # Perform vector similarity search with metadata filter
        logger.info("Performing vector similarity search...")
        metadata_filter = {
            "title": "Animal Spirits",
            "author": "Robert Schiller"
        }
        results = vector_db.search(query_embedding, metadata_filter=metadata_filter)
        
        if not results:
            logger.info("No relevant documents found")
            return []
            
        logger.info(f"Found {len(results)} relevant documents")
        
        # Process and return context
        context_parts = []
        for i, doc in enumerate(results, 1):
            logger.info(f"Processing document {i}/{len(results)}")
            logger.info(f"Document {i} source: {doc.metadata.get('source', 'Unknown')}")
            
            if not doc.text:
                logger.warning(f"Document {i} has no text content")
                continue
                
            context_parts.append(doc.text)
            
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
        flow = Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE,
            scopes=SCOPES,
            state=state,
            redirect_uri=GOOGLE_REDIRECT_URI
        )
        
        # Get the authorization response from the callback URL
        authorization_response = request.url
        flow.fetch_token(authorization_response=authorization_response)
        
        # Get credentials and user info
        credentials = flow.credentials
        id_info = id_token.verify_oauth2_token(
            credentials.id_token, 
            requests.Request(), 
            GOOGLE_CLIENT_ID
        )
        
        # Check if the email domain is allowed
        email = id_info['email']
        domain = email.split('@')[1]
        if domain not in ALLOWED_DOMAINS:
            logger.warning(f"Unauthorized domain: {domain}")
            session.clear()
            return redirect(url_for('login'))
            
        # Store user info in session
        session['user_id'] = email
        session['user_name'] = id_info.get('name', email)
        session['user_email'] = email
        session['credentials'] = {
            'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes
        }
        
        logger.info(f"User logged in successfully: {email}")
        return redirect(url_for('home'))
        
    except Exception as e:
        logger.error(f"Error in callback: {str(e)}")
        session.clear()
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    """Log out the user."""
    session.clear()
    logger.info("User logged out")
    return redirect(url_for('login'))

@app.route('/')
@login_required
def home():
    """Home page."""
    return render_template('home.html')

@app.route('/ingest', methods=['POST'])
@login_required
def ingest_content():
    """Ingest content from a directory."""
    try:
        data = request.get_json()
        if not data or 'directory' not in data:
            return jsonify({'error': 'No directory provided'}), 400
            
        directory = data['directory']
        if not os.path.isdir(directory):
            return jsonify({'error': 'Invalid directory path'}), 400
            
        # Process all PDF files in the directory
        pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
        processed_files = []
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(directory, pdf_file)
            try:
                book_id = process_book(pdf_path)
                if book_id:
                    processed_files.append(pdf_file)
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
                
        # Normalize author names after processing all books
        normalize_author_names()
        
        return jsonify({
            'message': f'Successfully processed {len(processed_files)} files',
            'processed_files': processed_files
        })
        
    except Exception as e:
        logger.error(f"Error in ingest_content: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    """Handle chat requests."""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
            
        user_message = data['message']
        logger.info(f"Received user message: {user_message}")
        
        # Get relevant context
        context_parts = get_relevant_context(user_message)
        
        if not context_parts:
            logger.info("No relevant context found - proceeding with general knowledge")
            
        # Prepare the context for the model
        context = "\n\n".join(context_parts)
        
        # Count tokens and truncate if necessary
        context_tokens = count_tokens(context)
        max_context_tokens = 4000  # Adjust based on model limits
        if context_tokens > max_context_tokens:
            context = truncate_text(context, max_context_tokens)
            
        # Prepare the messages for the model
        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides accurate information based on the provided context. If the context doesn't contain relevant information, say so and provide general knowledge."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_message}"}
        ]
        
        # Get response from OpenAI
        logger.info("Sending request to OpenAI with context")
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Extract and filter the response
        assistant_message = response.choices[0].message.content
        filtered_message = filter_sensitive_info(assistant_message)
        
        logger.info("Received response from OpenAI")
        return jsonify({
            'response': filtered_message,
            'context_used': bool(context_parts)
        })
        
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return jsonify({'error': str(e)}), 500

def truncate_text(text: str, max_tokens: int) -> str:
    """Truncate text to fit within token limit."""
    encoding = tiktoken.encoding_for_model("gpt-4-turbo-preview")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
        
    # Truncate to max_tokens
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

@app.route('/books')
@login_required
def get_books():
    """Get the list of books from the database."""
    try:
        # Query the database to get the list of books
        books = vector_db.get_books()  # Assuming there's a method to get books
        return jsonify({'books': books})
    except Exception as e:
        logger.error(f"Error retrieving books: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
=======
from flask import Flask, render_template, request, jsonify, send_file
from pathlib import Path
import os
from datetime import datetime
import json
from typing import Optional, Dict, Any

from src.processing.ingest_documents import ingest_documents
from src.database.postgres_vector_db import PostgreSQLVectorDB
from src.config.metadata_config import MetadataManager, Genre, SubGenre
from src.processing.url_processor import URLProcessor
from src.config.config import DB_CONFIG, DEFAULT_METADATA
from src.processing.rag_system import RAGSystem

app = Flask(__name__)

# Initialize RAG system
rag_system = RAGSystem()

def parse_metadata(metadata_str: Optional[str], template_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Parse metadata from form data and template."""
    metadata_manager = MetadataManager("src/config/metadata_templates.json")
    
    # Start with template if provided
    metadata = {}
    if template_name:
        try:
            metadata = metadata_manager.get_template(template_name)
        except ValueError as e:
            print(f"Warning: {e}")
    
    # Override with form metadata if provided
    if metadata_str:
        try:
            # Try to parse as JSON
            cmd_metadata = json.loads(metadata_str)
        except json.JSONDecodeError:
            # If not valid JSON, try to parse as key=value pairs
            cmd_metadata = {}
            for pair in metadata_str.split(','):
                if '=' not in pair:
                    continue
                key, value = pair.split('=', 1)
                cmd_metadata[key.strip()] = value.strip()
        
        metadata.update(cmd_metadata)
    
    # Enrich metadata with defaults
    metadata = metadata_manager.enrich_metadata(metadata)
    
    # Add default metadata
    metadata.update(DEFAULT_METADATA)
    
    # Validate metadata
    try:
        metadata_manager.validate_metadata(metadata)
    except ValueError as e:
        print(f"Warning: Invalid metadata: {e}")
    
    return metadata

@app.route('/')
def index():
    """Render the main interface."""
    return render_template('index.html')

@app.route('/ingest', methods=['POST'])
def ingest():
    """Handle document ingestion."""
    try:
        # Get form data
        file = request.files.get('file')
        url = request.form.get('url')
        template = request.form.get('template')
        subgenre = request.form.get('subgenre')
        topics = request.form.getlist('topics')
        year = request.form.get('year')
        metadata_str = request.form.get('metadata')
        
        # Parse metadata
        metadata = parse_metadata(metadata_str, template)
        
        # Override with form arguments
        if subgenre:
            metadata["subgenre"] = subgenre
        if topics:
            metadata["topics"] = topics
        if year:
            metadata["year"] = int(year)
        
        # Add ingestion metadata
        metadata.update({
            "ingestion_date": datetime.now().isoformat(),
            "ingested_by": "web_interface"
        })
        
        # Process based on input type
        if url:
            # Process URL
            url_processor = URLProcessor()
            doc = url_processor.process_url(url, metadata)
            doc_id = rag_system.db_manager.add_documents([doc])[0]
            return jsonify({
                "success": True,
                "message": f"Successfully ingested document from URL: {url}",
                "document_id": doc_id,
                "metadata": metadata
            })
        elif file:
            # Process file
            # Save uploaded file temporarily
            temp_path = Path("temp_uploads")
            temp_path.mkdir(exist_ok=True)
            file_path = temp_path / file.filename
            file.save(file_path)
            
            # Ingest document
            doc_ids = ingest_documents(
                str(file_path),
                rag_system.db_manager,
                metadata=metadata
            )
            
            # Clean up
            file_path.unlink()
            
            return jsonify({
                "success": True,
                "message": f"Successfully ingested document: {file.filename}",
                "document_ids": doc_ids,
                "metadata": metadata
            })
        else:
            return jsonify({
                "success": False,
                "message": "No file or URL provided"
            })
            
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        })

@app.route('/query', methods=['POST'])
def query():
    """Handle RAG queries."""
    try:
        query_text = request.form.get('query')
        k = int(request.form.get('k', 5))
        
        if not query_text:
            return jsonify({
                "success": False,
                "message": "No query provided"
            })
        
        # Process query
        response, results = rag_system.query(query_text, k)
        
        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.content,
                "metadata": doc.metadata,
                "score": float(score)
            })
        
        return jsonify({
            "success": True,
            "response": response,
            "results": formatted_results
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        })

@app.route('/documents', methods=['GET'])
def list_documents():
    """List all ingested documents."""
    try:
        db = PostgreSQLVectorDB()
        documents = db.get_all_documents()
        
        formatted_docs = []
        for doc in documents:
            formatted_docs.append({
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata
            })
        
        return jsonify({
            "success": True,
            "documents": formatted_docs
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        })

@app.route('/document/<doc_id>', methods=['GET'])
def get_document(doc_id):
    """Get a specific document by ID."""
    try:
        db = PostgreSQLVectorDB()
        doc = db.get_document(doc_id)
        
        if not doc:
            return jsonify({
                "success": False,
                "message": "Document not found"
            })
        
        return jsonify({
            "success": True,
            "document": {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    # Create index.html template
    index_template = templates_dir / "index.html"
    if not index_template.exists():
        with open(index_template, "w") as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>RAG Interface</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .container { max-width: 1200px; }
        .section { margin-bottom: 2rem; }
    </style>
</head>
<body>
    <div class="container mt-4">
        <h1 class="mb-4">RAG Interface</h1>
        
        <!-- Document Ingestion Section -->
        <div class="section">
            <h2>Ingest Documents</h2>
            <form id="ingestForm" class="mb-3">
                <div class="mb-3">
                    <label class="form-label">Upload File</label>
                    <input type="file" class="form-control" name="file">
                </div>
                <div class="mb-3">
                    <label class="form-label">Or Enter URL</label>
                    <input type="url" class="form-control" name="url">
                </div>
                <div class="mb-3">
                    <label class="form-label">Template</label>
                    <select class="form-select" name="template">
                        <option value="">None</option>
                        <option value="research_paper">Research Paper</option>
                        <option value="blog_post">Blog Post</option>
                        <option value="technical_doc">Technical Document</option>
                    </select>
                </div>
                <div class="mb-3">
                    <label class="form-label">Subgenre</label>
                    <select class="form-select" name="subgenre">
                        <option value="">None</option>
                        <option value="machine_learning">Machine Learning</option>
                        <option value="data_science">Data Science</option>
                        <option value="other">Other</option>
                    </select>
                </div>
                <div class="mb-3">
                    <label class="form-label">Topics</label>
                    <input type="text" class="form-control" name="topics" placeholder="Comma-separated topics">
                </div>
                <div class="mb-3">
                    <label class="form-label">Year</label>
                    <input type="number" class="form-control" name="year">
                </div>
                <button type="submit" class="btn btn-primary">Ingest</button>
            </form>
            <div id="ingestResult" class="alert" style="display: none;"></div>
        </div>
        
        <!-- Query Section -->
        <div class="section">
            <h2>Query Documents</h2>
            <form id="queryForm" class="mb-3">
                <div class="mb-3">
                    <label class="form-label">Query</label>
                    <input type="text" class="form-control" name="query" required>
                </div>
                <div class="mb-3">
                    <label class="form-label">Number of Results</label>
                    <input type="number" class="form-control" name="k" value="5" min="1" max="20">
                </div>
                <button type="submit" class="btn btn-primary">Query</button>
            </form>
            <div id="queryResult" class="alert" style="display: none;"></div>
        </div>
        
        <!-- Document List Section -->
        <div class="section">
            <h2>Ingested Documents</h2>
            <div id="documentList" class="list-group"></div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Handle document ingestion
        document.getElementById('ingestForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            
            try {
                const response = await fetch('/ingest', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                
                const resultDiv = document.getElementById('ingestResult');
                resultDiv.textContent = result.message;
                resultDiv.className = `alert alert-${result.success ? 'success' : 'danger'}`;
                resultDiv.style.display = 'block';
                
                if (result.success) {
                    loadDocuments();
                }
            } catch (error) {
                console.error('Error:', error);
                const resultDiv = document.getElementById('ingestResult');
                resultDiv.textContent = 'An error occurred';
                resultDiv.className = 'alert alert-danger';
                resultDiv.style.display = 'block';
            }
        });
        
        // Handle queries
        document.getElementById('queryForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            
            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                
                const resultDiv = document.getElementById('queryResult');
                if (result.success) {
                    let html = `<h4>Response:</h4><p>${result.response}</p>`;
                    html += `<h4>Retrieved Documents:</h4>`;
                    result.results.forEach((doc, i) => {
                        html += `
                            <div class="card mb-2">
                                <div class="card-body">
                                    <h5 class="card-title">Document ${i + 1} (Score: ${doc.score.toFixed(3)})</h5>
                                    <p class="card-text">${doc.content.substring(0, 200)}...</p>
                                </div>
                            </div>
                        `;
                    });
                    resultDiv.innerHTML = html;
                } else {
                    resultDiv.textContent = result.message;
                }
                resultDiv.className = `alert alert-${result.success ? 'success' : 'danger'}`;
                resultDiv.style.display = 'block';
            } catch (error) {
                console.error('Error:', error);
                const resultDiv = document.getElementById('queryResult');
                resultDiv.textContent = 'An error occurred';
                resultDiv.className = 'alert alert-danger';
                resultDiv.style.display = 'block';
            }
        });
        
        // Load and display documents
        async function loadDocuments() {
            try {
                const response = await fetch('/documents');
                const result = await response.json();
                
                if (result.success) {
                    const documentList = document.getElementById('documentList');
                    documentList.innerHTML = result.documents.map(doc => `
                        <div class="list-group-item">
                            <h5 class="mb-1">Document ${doc.id}</h5>
                            <p class="mb-1">${doc.content.substring(0, 100)}...</p>
                            <small>Topics: ${doc.metadata.topics.join(', ')}</small>
                        </div>
                    `).join('');
                }
            } catch (error) {
                console.error('Error:', error);
            }
        }
        
        // Load documents on page load
        loadDocuments();
    </script>
</body>
</html>
            """)
    
    app.run(debug=True, port=5009) 
>>>>>>> 8f39c0cbc19721b9785a7f78d10722be3f0eb339
