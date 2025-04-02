from flask import session, redirect, url_for
from functools import wraps

class SessionManager:
    @staticmethod
    def login_required(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user' not in session:
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return decorated_function
        
    @staticmethod
    def set_user(user_info):
        """Set user information in the session"""
        session['user'] = {
            'email': user_info['email'],
            'name': user_info.get('name', ''),
            'picture': user_info.get('picture', '')
        }
        
    @staticmethod
    def get_user():
        """Get current user information from session"""
        return session.get('user')
        
    @staticmethod
    def clear():
        """Clear the session"""
        session.clear() 