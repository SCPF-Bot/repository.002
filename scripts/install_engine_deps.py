#!/usr/bin/env python3
import subprocess
import sys
import argparse
import logging
import importlib.util
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("JIT_Installer")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
REQUIREMENTS_DIR = REPO_ROOT / "requirements"

OCR_CONFIG = {
    "google_vision": ("ocr_google_vision.txt", ["google-cloud-vision"], "google"),
    "manga_ocr": ("ocr_manga_ocr.txt", ["manga-ocr"], "manga_ocr"),
    "paddle_ocr": ("ocr_paddle_ocr.txt", ["paddlepaddle", "paddleocr"], "paddleocr"),
    "tesseract": (None, None, "pytesseract"),
}

TTS_CONFIG = {
    "elevenlabs": ("tts_elevenlabs.txt", ["elevenlabs"], "elevenlabs"),
    "edge_tts": ("tts_edge_tts.txt", ["edge-tts"], "edge_tts"),
    "xtts_v2": ("tts_xtts_v2.txt", ["TTS"], "TTS"),
    "melo_tts": ("tts_melo_tts.txt", ["mecab-python3", "git+https://github.com/myshell-ai/MeloTTS.git"], "melo"),
}

def check_package(package_name: str) -> bool:
    """Check if a package is already installed safely."""
    if not package_name:
        return True
    try:
        # Check only the top-level package to avoid Namespace errors
        top_level = package_name.split('.')[0]
        return importlib.util.find_spec(top_level) is not None
    except (ModuleNotFoundError, AttributeError, ValueError):
        return False

def install_deps(req_filename: str, fallback_pkgs: list, import_name: str):
    if check_package(import_name):
        logger.info(f"✅ {import_name} is already installed. Skipping.")
        return

    req_path = REQUIREMENTS_DIR / req_filename if req_filename else None
    base_cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--prefer-binary"]

    try:
        if req_path and req_path.exists():
            logger.info(f"Installing from {req_path}...")
            subprocess.check_call(base_cmd + ["-r", str(req_path)])
        elif fallback_pkgs:
            logger.info(f"Installing {fallback_pkgs} via fallback...")
            subprocess.check_call(base_cmd + fallback_pkgs)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install {import_name}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr", required=True, choices=OCR_CONFIG.keys())
    parser.add_argument("--tts", required=True, choices=TTS_CONFIG.keys())
    args = parser.parse_args()

    ocr_file, ocr_fallback, ocr_import = OCR_CONFIG[args.ocr]
    install_deps(ocr_file, ocr_fallback, ocr_import)

    tts_file, tts_fallback, tts_import = TTS_CONFIG[args.tts]
    install_deps(tts_file, tts_fallback, tts_import)

if __name__ == "__main__":
    main()
