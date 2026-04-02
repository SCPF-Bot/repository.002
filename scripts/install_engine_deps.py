#!/usr/bin/env python3
"""
Just-in-Time installation of OCR/TTS engine dependencies.
Uses absolute paths to requirement files.
"""
import subprocess
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("JIT_Installer")

# Get absolute paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
REQUIREMENTS_DIR = REPO_ROOT / "requirements"

# Mapping engine names to either a requirement file or a direct pip package
OCR_CONFIG = {
    "google_vision": ("ocr_google_vision.txt", "google-cloud-vision"),
    "manga_ocr": ("ocr_manga_ocr.txt", "manga-ocr"),
    "paddle_ocr": ("ocr_paddle_ocr.txt", "paddlepaddle paddleocr"),
    "tesseract": (None, None),  # system package, installed via system_deps.sh
}

TTS_CONFIG = {
    "elevenlabs": ("tts_elevenlabs.txt", None),   # no extra pip package needed
    "edge_tts": ("tts_edge_tts.txt", "edge-tts"),
    "xtts_v2": ("tts_xtts_v2.txt", "TTS"),
    "melo_tts": ("tts_melo_tts.txt", "git+https://github.com/myshell-ai/MeloTTS.git"),
}

def install_from_file_or_fallback(req_filename: str, fallback_pkg: str = None):
    """Install dependencies from a requirements file, or directly if file missing."""
    if req_filename is None:
        return
    req_path = REQUIREMENTS_DIR / req_filename
    if req_path.exists():
        logger.info(f"Installing from {req_path}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_path)])
    elif fallback_pkg:
        logger.warning(f"Requirement file {req_path} not found. Installing {fallback_pkg} directly.")
        if fallback_pkg.startswith("git+"):
            subprocess.check_call([sys.executable, "-m", "pip", "install", fallback_pkg])
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + fallback_pkg.split())
    else:
        logger.info(f"No Python dependencies required for this engine (system package or no extra deps).")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr", required=True, choices=OCR_CONFIG.keys())
    parser.add_argument("--tts", required=True, choices=TTS_CONFIG.keys())
    args = parser.parse_args()

    # Install OCR dependencies
    ocr_file, ocr_fallback = OCR_CONFIG[args.ocr]
    install_from_file_or_fallback(ocr_file, ocr_fallback)

    # Install TTS dependencies
    tts_file, tts_fallback = TTS_CONFIG[args.tts]
    install_from_file_or_fallback(tts_file, tts_fallback)

if __name__ == "__main__":
    main()
