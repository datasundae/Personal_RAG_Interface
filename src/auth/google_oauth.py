import os
import json
from flask import redirect, session, url_for
from google.oauth2 import id_token
from google.auth.transport import requests
from google_auth_oauthlib.flow import Flow

class GoogleOAuth:
    def __init__(self, client_secret_path: str, scopes: list):
        self.client_secret_path = client_secret_path
        self.scopes = scopes
        self.flow = None
        
    def init_app(self, app):
        """Initialize the OAuth flow with Flask app configuration"""
        self.flow = Flow.from_client_secrets_file(
            self.client_secret_path,
            scopes=self.scopes,
            redirect_uri=url_for('auth_callback', _external=True)
        )
        
    def get_authorization_url(self):
        """Get the authorization URL for Google OAuth"""
        authorization_url, state = self.flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true'
        )
        session['state'] = state
        return authorization_url
        
    def fetch_token(self, auth_response):
        """Fetch the token from Google OAuth response"""
        self.flow.fetch_token(authorization_response=auth_response)
        credentials = self.flow.credentials
        return credentials
        
    def verify_token(self, token):
        """Verify the ID token from Google"""
        try:
            idinfo = id_token.verify_oauth2_token(
                token, requests.Request(), os.getenv('GOOGLE_CLIENT_ID'))
            return idinfo
        except Exception as e:
            print(f"Token verification failed: {e}")
            return None 