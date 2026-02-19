# Use Python 3.14 as base image
FROM python:3.14-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    QT_QPA_PLATFORM=offscreen \
    DISPLAY=:99

# Install system dependencies required by some Python packages
# tesseract-ocr for pytesseract, portaudio19-dev for PyAudio
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    portaudio19-dev \
    gcc \
    g++ \
    make \
    xvfb \
    x11-utils \
    x11-xserver-utils \
    xorg \
    xserver-xorg-video-dummy \
    libx11-dev \
    libxtst-dev \
    libxrandr-dev \
    python3-tk \
    python3-dev \
    tk-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create volume mount point for persistent data
VOLUME ["/app/data"]

# Create startup script to launch Xvfb and the application
RUN echo '#!/bin/bash\n\
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &\n\
sleep 1\n\
exec python Echo.py "$@"' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Set entry point
ENTRYPOINT ["/app/entrypoint.sh"]
