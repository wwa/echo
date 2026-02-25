#!/bin/bash
# Start script for vulnerable fake bank application

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./setup.sh first"
    exit 1
fi

# Check if database exists
if [ ! -f "bank.db" ]; then
    echo "❌ Database not found!"
    echo "Please run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Start the application
echo "🚀 Starting Vulnerable Fake Bank..."
echo ""
python3 app.py
