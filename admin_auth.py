"""
Admin Authentication Module
Handles login, session management, and security for portfolio admin panel
"""

from functools import wraps
from flask import session, redirect, url_for, flash
from werkzeug.security import check_password_hash, generate_password_hash
import os
import time
from datetime import datetime

# Admin password (must be set in .env file)
ADMIN_PASSWORD_HASH = os.getenv('ADMIN_PASSWORD_HASH')
if not ADMIN_PASSWORD_HASH:
    raise ValueError("ADMIN_PASSWORD_HASH must be set in .env file")

def login_required(f):
    """Decorator to require admin login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            flash('Please log in to access the admin panel.', 'error')
            return redirect(url_for('admin_login'))
        
        # Check session timeout (30 minutes)
        last_activity = session.get('last_activity')
        if last_activity:
            # last_activity is a timestamp (float)
            if time.time() - last_activity > 1800:  # 1800 seconds = 30 minutes
                session.clear()
                flash('Session expired. Please log in again.', 'warning')
                return redirect(url_for('admin_login'))
        
        # Update last activity
        session['last_activity'] = time.time()
        return f(*args, **kwargs)
    return decorated_function

def verify_password(password):
    """Verify admin password"""
    # In production, use environment variable
    # admin_hash = os.getenv('ADMIN_PASSWORD_HASH')
    return check_password_hash(ADMIN_PASSWORD_HASH, password)

def log_admin_action(action, details=""):
    """Log admin actions for audit trail"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {action}"
    if details:
        log_entry += f" - {details}"
    
    # Append to log file
    log_file = os.path.join(os.path.dirname(__file__), 'admin_log.txt')
    with open(log_file, 'a') as f:
        f.write(log_entry + '\n')
