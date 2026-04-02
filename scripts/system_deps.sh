#!/bin/bash
set -e

echo "=== Installing Modernized System Dependencies ==="

# Update package lists
sudo apt-get update

# Install dependencies
# Note: libgl1 replaces the obsolete libgl1-mesa-glx
sudo apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    unzip \
    curl \
    espeak-ng \
    libgomp1

# Cleanup to keep the runner image lean
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*

echo "✓ System dependencies installed successfully"
