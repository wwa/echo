#!/usr/bin/env python3
"""
VULNERABLE FAKE BANK APPLICATION
================================
WARNING: This application contains intentional security vulnerabilities
for educational and testing purposes only. DO NOT use in production!

Vulnerabilities included:
1. SQL Injection in login and transaction search
2. Cross-Site Scripting (XSS) in user messages and transaction notes
3. No CSRF protection
4. Weak session management
"""

from flask import Flask, request, render_template_string, session, redirect, url_for
import sqlite3
import os
from datetime import datetime
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Configuration from environment
HOST = os.getenv('HOST', '127.0.0.1')
PORT = int(os.getenv('PORT', 8200))
DEBUG = os.getenv('DEBUG', 'true').lower() == 'true'
DB_PATH = os.getenv('DB_PATH', 'bank.db')
SECRET_KEY = os.getenv('SECRET_KEY', 'insecure_secret_key_12345')
SESSION_COOKIE_NAME = os.getenv('SESSION_COOKIE_NAME', 'bank_session')
SESSION_TIMEOUT = int(os.getenv('SESSION_TIMEOUT', 3600))
MAX_TRANSACTIONS = int(os.getenv('MAX_TRANSACTIONS', 10))
MAX_MESSAGES = int(os.getenv('MAX_MESSAGES', 5))
SHOW_DETAILED_ERRORS = os.getenv('SHOW_DETAILED_ERRORS', 'true').lower() == 'true'
SHOW_VULNERABILITY_HINTS = os.getenv('SHOW_VULNERABILITY_HINTS', 'true').lower() == 'true'
SHOW_TEST_ACCOUNTS = os.getenv('SHOW_TEST_ACCOUNTS', 'true').lower() == 'true'
ENABLE_REQUEST_LOGGING = os.getenv('ENABLE_REQUEST_LOGGING', 'true').lower() == 'true'
LOG_FILE = os.getenv('LOG_FILE', 'bank_access.log')

# Configure logging
if ENABLE_REQUEST_LOGGING:
    # Create a separate logger for application logs to avoid conflicts with Flask/Werkzeug
    app_logger = logging.getLogger('bank_app')
    app_logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_FILE)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    app_logger.addHandler(handler)
else:
    app_logger = None

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['SESSION_COOKIE_NAME'] = SESSION_COOKIE_NAME
app.config['PERMANENT_SESSION_LIFETIME'] = SESSION_TIMEOUT

# HTML Templates with intentional XSS vulnerabilities
LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SecureBank Login</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 400px; margin: 50px auto; padding: 20px; }
        .container { border: 1px solid #ddd; padding: 20px; border-radius: 5px; }
        input { width: 100%; padding: 10px; margin: 10px 0; box-sizing: border-box; }
        button { width: 100%; padding: 10px; background: #007bff; color: white; border: none; cursor: pointer; }
        button:hover { background: #0056b3; }
        .error { color: red; padding: 10px; }
        .info { color: #666; font-size: 12px; margin-top: 20px; }
        .hint { background: #fff3cd; padding: 10px; margin-top: 10px; font-size: 11px; border-left: 3px solid #ffc107; }
    </style>
</head>
<body>
    <div class="container">
        <h2>🏦 SecureBank Login</h2>
        {% if error %}
        <div class="error">{{ error|safe }}</div>
        {% endif %}
        <form method="POST">
            <input type="text" name="username" placeholder="Username" required>
            <input type="password" name="password" placeholder="Password" required>
            <button type="submit">Login</button>
        </form>
        {% if show_test_accounts %}
        <div class="info">
            <strong>Test Accounts:</strong><br>
            admin / admin123<br>
            john.doe / password<br>
            jane.smith / qwerty
        </div>
        {% endif %}
        {% if show_hints %}
        <div class="hint">
            <strong>⚠️ Vulnerability Hint:</strong> This login form is vulnerable to SQL injection.<br>
            Try: <code>admin' --</code> or <code>' OR '1'='1</code>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SecureBank - Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        .header { background: #007bff; color: white; padding: 20px; margin: -20px -20px 20px -20px; }
        .header h1 { margin: 0; }
        .logout { float: right; color: white; text-decoration: none; }
        .balance { background: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }
        .section { margin: 30px 0; }
        input, textarea { padding: 10px; margin: 5px 0; width: 300px; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
        button:hover { background: #0056b3; }
        .transaction { border-bottom: 1px solid #ddd; padding: 10px 0; }
        .message { background: #fff3cd; padding: 10px; margin: 10px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <a href="/logout" class="logout">Logout</a>
        <h1>🏦 SecureBank</h1>
        <p>Welcome, {{ username|safe }}</p>
    </div>

    <div class="balance">
        <h2>Account Balance</h2>
        <h1>${{ balance }}</h1>
    </div>

    <div class="section">
        <h3>💬 Leave a Message</h3>
        <form method="POST" action="/message">
            <textarea name="message" placeholder="Your message here..." rows="3"></textarea><br>
            <button type="submit">Post Message</button>
        </form>

        <h4>Recent Messages:</h4>
        {% for msg in messages %}
        <div class="message">
            <strong>{{ msg[0]|safe }}</strong>: {{ msg[1]|safe }}
        </div>
        {% endfor %}
    </div>

    <div class="section">
        <h3>🔍 Search Transactions</h3>
        <form method="POST" action="/search">
            <input type="text" name="query" placeholder="Search by description...">
            <button type="submit">Search</button>
        </form>

        {% if search_results %}
        <h4>Search Results:</h4>
        {% for tx in search_results %}
        <div class="transaction">
            <strong>{{ tx[0] }}</strong> - ${{ tx[1] }} - {{ tx[2]|safe }}
        </div>
        {% endfor %}
        {% endif %}
    </div>

    <div class="section">
        <h3>📊 Recent Transactions</h3>
        {% for tx in transactions %}
        <div class="transaction">
            <strong>{{ tx[0] }}</strong> - ${{ tx[1] }} - {{ tx[2] }}
        </div>
        {% endfor %}
    </div>
</body>
</html>
"""

def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    return conn

@app.route('/', methods=['GET', 'POST'])
def login():
    """VULNERABLE: SQL Injection in login"""
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')

        # Log login attempt
        if ENABLE_REQUEST_LOGGING and app_logger:
            app_logger.info(f"Login attempt - Username: {username} from {request.remote_addr}")

        # VULNERABILITY: SQL Injection - no parameterization
        conn = get_db()
        cursor = conn.cursor()
        query = f"SELECT id, username, balance FROM users WHERE username = '{username}' AND password = '{password}'"

        try:
            cursor.execute(query)
            user = cursor.fetchone()
            conn.close()

            if user:
                session['user_id'] = user[0]
                session['username'] = user[1]
                if ENABLE_REQUEST_LOGGING and app_logger:
                    app_logger.info(f"Login successful - User ID: {user[0]}, Username: {user[1]}")
                return redirect(url_for('dashboard'))
            else:
                if ENABLE_REQUEST_LOGGING and app_logger:
                    app_logger.warning(f"Login failed - Invalid credentials for username: {username}")
                return render_template_string(LOGIN_TEMPLATE, error="Invalid credentials", show_hints=SHOW_VULNERABILITY_HINTS, show_test_accounts=SHOW_TEST_ACCOUNTS)
        except Exception as e:
            conn.close()
            # VULNERABILITY: Error messages reveal SQL structure
            if ENABLE_REQUEST_LOGGING and app_logger:
                app_logger.error(f"Login error - SQL Exception: {str(e)}")
            error_msg = f"Database error: {str(e)}" if SHOW_DETAILED_ERRORS else "An error occurred. Please try again."
            return render_template_string(LOGIN_TEMPLATE, error=error_msg, show_hints=SHOW_VULNERABILITY_HINTS, show_test_accounts=SHOW_TEST_ACCOUNTS)

    return render_template_string(LOGIN_TEMPLATE, show_hints=SHOW_VULNERABILITY_HINTS, show_test_accounts=SHOW_TEST_ACCOUNTS)

@app.route('/dashboard')
def dashboard():
    """Dashboard page with user info"""
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_db()
    cursor = conn.cursor()

    # Get user balance
    cursor.execute("SELECT balance FROM users WHERE id = ?", (session['user_id'],))
    balance = cursor.fetchone()[0]

    # Get recent transactions (using MAX_TRANSACTIONS from config)
    cursor.execute("""
        SELECT date, amount, description
        FROM transactions
        WHERE user_id = ?
        ORDER BY date DESC LIMIT ?
    """, (session['user_id'], MAX_TRANSACTIONS))
    transactions = cursor.fetchall()

    # Get messages (using MAX_MESSAGES from config)
    cursor.execute("""
        SELECT username, message
        FROM messages
        ORDER BY id DESC LIMIT ?
    """, (MAX_MESSAGES,))
    messages = cursor.fetchall()

    conn.close()

    # VULNERABILITY: XSS - username and messages not escaped
    return render_template_string(
        DASHBOARD_TEMPLATE,
        username=session['username'],
        balance=balance,
        transactions=transactions,
        messages=messages
    )

@app.route('/message', methods=['POST'])
def post_message():
    """VULNERABLE: XSS in messages"""
    if 'user_id' not in session:
        return redirect(url_for('login'))

    message = request.form.get('message', '')

    # VULNERABILITY: No XSS sanitization
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (username, message) VALUES (?, ?)",
        (session['username'], message)
    )
    conn.commit()
    conn.close()

    return redirect(url_for('dashboard'))

@app.route('/search', methods=['POST'])
def search_transactions():
    """VULNERABLE: SQL Injection in search"""
    if 'user_id' not in session:
        return redirect(url_for('login'))

    query = request.form.get('query', '')

    # Log search attempt
    if ENABLE_REQUEST_LOGGING and app_logger:
        app_logger.info(f"Transaction search - User: {session['username']}, Query: {query}")

    conn = get_db()
    cursor = conn.cursor()

    # Get user balance
    cursor.execute("SELECT balance FROM users WHERE id = ?", (session['user_id'],))
    balance = cursor.fetchone()[0]

    # Get recent transactions (using MAX_TRANSACTIONS from config)
    cursor.execute("""
        SELECT date, amount, description
        FROM transactions
        WHERE user_id = ?
        ORDER BY date DESC LIMIT ?
    """, (session['user_id'], MAX_TRANSACTIONS))
    transactions = cursor.fetchall()

    # Get messages (using MAX_MESSAGES from config)
    cursor.execute("""
        SELECT username, message
        FROM messages
        ORDER BY id DESC LIMIT ?
    """, (MAX_MESSAGES,))
    messages = cursor.fetchall()

    # VULNERABILITY: SQL Injection in search
    search_query = f"SELECT date, amount, description FROM transactions WHERE user_id = {session['user_id']} AND description LIKE '%{query}%'"

    try:
        cursor.execute(search_query)
        search_results = cursor.fetchall()
        if ENABLE_REQUEST_LOGGING and app_logger:
            app_logger.info(f"Search successful - Found {len(search_results)} results")
    except Exception as e:
        if ENABLE_REQUEST_LOGGING and app_logger:
            app_logger.error(f"Search error - SQL Exception: {str(e)}")
        error_msg = str(e) if SHOW_DETAILED_ERRORS else "Search error"
        search_results = [(error_msg, 0, "Error")]

    conn.close()

    return render_template_string(
        DASHBOARD_TEMPLATE,
        username=session['username'],
        balance=balance,
        transactions=transactions,
        messages=messages,
        search_results=search_results
    )

@app.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    print("=" * 60)
    print("VULNERABLE FAKE BANK APPLICATION")
    print("=" * 60)
    print("WARNING: Contains intentional vulnerabilities!")
    print("For educational and testing purposes only.")
    print("\nVulnerabilities:")
    print("  1. SQL Injection in login (username/password)")
    print("  2. SQL Injection in transaction search")
    print("  3. XSS in user messages")
    print("  4. XSS in transaction notes")
    print("\nConfiguration:")
    print(f"  Host: {HOST}")
    print(f"  Port: {PORT}")
    print(f"  Debug: {DEBUG}")
    print(f"  Database: {DB_PATH}")
    print(f"  Request Logging: {ENABLE_REQUEST_LOGGING}")
    if ENABLE_REQUEST_LOGGING:
        print(f"  Log File: {LOG_FILE}")
    print(f"\nStarting server on http://{HOST}:{PORT}")
    print("=" * 60)
    app.run(debug=DEBUG, host=HOST, port=PORT)
