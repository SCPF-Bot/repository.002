#!/bin/bash
set -e

echo "=== Installing Optimized System Dependencies ==="
sudo apt-get update

sudo apt-get install -y --no-install-recommends \
    ffmpeg \
    ffprobe \
    tesseract-ocr \
    libtesseract-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    unzip \
    curl \
    espeak-ng

sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*
echo "✓ System dependencies ready"
