import unittest
from flask import Flask
from src.web.app import app
from src.database.db_connection import db
import os
from pathlib import Path

class TestWebInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.app = app.test_client()
        cls.app.testing = True
        cls.project_root = str(Path(__file__).parent.parent)

    def setUp(self):
        """Set up before each test."""
        self.app = app.test_client()
        self.app.testing = True

    def test_home_page(self):
        """Test that the home page loads."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 302)  # Should redirect to login

    def test_login_page(self):
        """Test that the login page loads."""
        response = self.app.get('/login')
        self.assertEqual(response.status_code, 200)

    def test_database_connection(self):
        """Test database connection."""
        try:
            conn = db.get_connection()
            self.assertIsNotNone(conn)
            self.assertFalse(conn.closed)
        except Exception as e:
            self.fail(f"Database connection failed: {str(e)}")

    def test_config_files_exist(self):
        """Test that all required configuration files exist."""
        required_files = [
            os.path.join(self.project_root, '.env'),
            os.path.join(self.project_root, 'src', 'web', 'config.py'),
            os.path.join(self.project_root, 'src', 'web', 'templates', 'login.html'),
            os.path.join(self.project_root, 'src', 'web', 'templates', 'home.html'),
            os.path.join(self.project_root, 'src', 'web', 'templates', 'index.html')
        ]
        
        for file_path in required_files:
            self.assertTrue(os.path.exists(file_path), f"Required file not found: {file_path}")

    def test_session_directory(self):
        """Test that the session directory exists and is writable."""
        session_dir = os.path.join(self.project_root, 'flask_session')
        self.assertTrue(os.path.exists(session_dir))
        self.assertTrue(os.access(session_dir, os.W_OK))

    def test_static_files(self):
        """Test that static files directory exists."""
        static_dir = os.path.join(self.project_root, 'src', 'web', 'static')
        self.assertTrue(os.path.exists(static_dir))

    def test_data_directories(self):
        """Test that all required data directories exist."""
        required_dirs = [
            os.path.join(self.project_root, 'data'),
            os.path.join(self.project_root, 'data', 'books'),
            os.path.join(self.project_root, 'data', 'art'),
            os.path.join(self.project_root, 'data', 'files')
        ]
        
        for dir_path in required_dirs:
            self.assertTrue(os.path.exists(dir_path), f"Required directory not found: {dir_path}")

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        if hasattr(cls, 'db'):
            cls.db.close()

if __name__ == '__main__':
    unittest.main() 