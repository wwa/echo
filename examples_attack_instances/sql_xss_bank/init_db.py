#!/usr/bin/env python3
"""
Database initialization script for vulnerable fake bank
Creates tables and populates with test data
"""

import sqlite3
from datetime import datetime, timedelta
import random

DB_PATH = 'bank.db'

def init_database():
    """Initialize database with schema and test data"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT,
            balance REAL DEFAULT 0.0
        )
    """)

    # Create transactions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            amount REAL NOT NULL,
            description TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Create messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Insert test users
    users = [
        ('admin', 'admin123', 'admin@securebank.local', 50000.00),
        ('john.doe', 'password', 'john.doe@email.com', 15234.50),
        ('jane.smith', 'qwerty', 'jane.smith@email.com', 8750.25),
        ('bob.wilson', '123456', 'bob.wilson@email.com', 3421.00),
        ('alice.jones', 'password123', 'alice.jones@email.com', 12890.75)
    ]

    for username, password, email, balance in users:
        try:
            cursor.execute(
                "INSERT INTO users (username, password, email, balance) VALUES (?, ?, ?, ?)",
                (username, password, email, balance)
            )
        except sqlite3.IntegrityError:
            print(f"User {username} already exists, skipping...")

    # Insert test transactions
    transaction_types = [
        ("Salary Deposit", 2500, 3500),
        ("ATM Withdrawal", -50, -200),
        ("Online Purchase", -20, -150),
        ("Restaurant Payment", -15, -80),
        ("Utility Bill", -50, -200),
        ("Transfer from Friend", 50, 300),
        ("Grocery Shopping", -40, -120),
        ("Gas Station", -30, -60),
        ("Coffee Shop", -5, -15),
        ("Movie Tickets", -20, -40)
    ]

    for user_id in range(1, 6):
        for i in range(20):
            tx_type, min_amt, max_amt = random.choice(transaction_types)
            amount = round(random.uniform(min_amt, max_amt), 2)
            date = (datetime.now() - timedelta(days=random.randint(0, 90))).strftime("%Y-%m-%d %H:%M:%S")

            cursor.execute(
                "INSERT INTO transactions (user_id, date, amount, description) VALUES (?, ?, ?, ?)",
                (user_id, date, amount, tx_type)
            )

    # Insert test messages (including some with potential XSS)
    messages = [
        ('admin', 'Welcome to SecureBank! We value your security.'),
        ('john.doe', 'Great service! Love the new mobile app.'),
        ('jane.smith', 'Quick question about my account balance.'),
        ('bob.wilson', 'Thanks for the quick support response!'),
        ('alice.jones', 'The online banking system is very convenient.')
    ]

    for username, message in messages:
        cursor.execute(
            "INSERT INTO messages (username, message) VALUES (?, ?)",
            (username, message)
        )

    conn.commit()
    conn.close()

    print("✅ Database initialized successfully!")
    print("\nTest Users:")
    print("-" * 50)
    for username, password, email, balance in users:
        print(f"  Username: {username:15} Password: {password:12} Balance: ${balance:,.2f}")
    print("-" * 50)
    print("\n💡 SQL Injection Test Payloads:")
    print("  Username: admin' --")
    print("  Password: (anything)")
    print("\n  Username: ' OR '1'='1")
    print("  Password: ' OR '1'='1")
    print("\n💡 XSS Test Payloads:")
    print("  Message: <script>alert('XSS')</script>")
    print("  Message: <img src=x onerror=alert('XSS')>")
    print("\n⚠️  Use these payloads responsibly for testing only!")

if __name__ == '__main__':
    print("Initializing vulnerable fake bank database...")
    init_database()
