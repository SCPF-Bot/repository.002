import os
import re
import zipfile
import requests
import shutil
from pathlib import Path
from natsort import natsorted
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def download_file(url: str, dest: Path) -> None:
    """Download a file with streaming and resume support."""
    logger.info(f"Downloading {url} -> {dest}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

def extract_archive(archive_path: Path, extract_to: Path) -> list:
    """Extract a ZIP/CBZ archive and return sorted list of image paths."""
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    # Find all images
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    images = [p for p in extract_to.rglob('*') if p.suffix.lower() in image_extensions]
    return natsorted(images)

def resize_and_pad(image_path: Path, output_path: Path, target_size=(1920, 1080)) -> None:
    """Resize image to fit target size, adding black bars (letterbox)."""
    img = Image.open(image_path).convert('RGB')
    target_w, target_h = target_size
    ratio = min(target_w / img.width, target_h / img.height)
    new_w = int(img.width * ratio)
    new_h = int(img.height * ratio)
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    new_img = Image.new('RGB', target_size, (0, 0, 0))
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
    new_img.paste(img_resized, offset)
    new_img.save(output_path, 'JPEG', quality=90)

def get_audio_duration(audio_path: Path) -> float:
    """Return duration in seconds using ffprobe."""
    import subprocess
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())

def cleanup_temp_dirs(*paths):
    for p in paths:
        try:
            shutil.rmtree(p)
        except Exception:
            pass
