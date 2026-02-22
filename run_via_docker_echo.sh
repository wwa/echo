#!/bin/bash

# Simple ECHO Docker Runner
# Builds/runs container and starts Echo.py interactively

set -e

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    exit 1
fi

# Parse build flag
BUILD_FLAG=""
if [ "$1" == "--build" ] || [ "$1" == "-b" ]; then
    BUILD_FLAG="--build"
    echo "Building Docker image..."
fi

# Create directories
mkdir -p data logs

# Build image if needed
if [ -n "$BUILD_FLAG" ]; then
    docker build -t echo-v1:latest -f docker/Dockerfile .
fi

# Run container with Echo.py
echo "Starting ECHO assistant..."
docker run --rm -it \
    --name echo-assistant \
    -v "$(pwd)/.env:/app/.env:ro" \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/logs:/app/logs" \
    -e DISPLAY=:99 \
    echo-v1:latest
