#!/bin/bash
set -e

echo "--- Initializing System Dependency Injection ---"

# 1. Update and install core binaries
sudo apt-get update
sudo apt-get install -y \
    ffmpeg \
    tesseract-ocr \
    libtesseract-dev \
    libmagic1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    unzip \
    curl

# 2. Clean up apt cache to save runner disk space
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*

# 3. Ensure the output and processing directories exist with proper permissions
mkdir -p output processing scripts
chmod +x scripts/*.sh

echo "--- System Dependencies Verified and Installed ---"
