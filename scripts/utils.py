import os
import shutil
import asyncio
import zipfile
import logging
from pathlib import Path
from typing import List, Tuple, Set

import aiohttp
import aiofiles
from PIL import Image
from natsort import natsorted

logger = logging.getLogger(__name__)

async def download_file(url: str, dest: Path) -> None:
    timeout = aiohttp.ClientTimeout(total=600) 
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, allow_redirects=True) as resp:
            resp.raise_for_status()
            async with aiofiles.open(dest, 'wb') as f:
                async for chunk in resp.content.iter_chunked(16384):
                    await f.write(chunk)

def _sync_extract(archive_path: Path, extract_to: Path) -> List[Path]:
    image_extensions: Set[str] = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    extracted_images = []
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        for member in zip_ref.infolist():
            if member.is_dir() or "__MACOSX" in member.filename or member.filename.startswith('.'):
                continue
            if Path(member.filename).suffix.lower() in image_extensions:
                zip_ref.extract(member, extract_to)
                extracted_images.append(extract_to / member.filename)
    return natsorted(extracted_images)

async def extract_archive(archive_path: Path, extract_to: Path) -> List[Path]:
    extract_to.mkdir(parents=True, exist_ok=True)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_extract, archive_path, extract_to)

def resize_and_pad(image_path: Path, output_path: Path, target_size: Tuple[int, int] = (1080, 1920)) -> None:
    """
    Optimized for Mobile Portrait (9:16). 
    Resizes manga page to fit height and pads width with black bars.
    """
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        
        # Scale image to fit within the 1080x1920 box while maintaining aspect ratio
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Create a vertical black canvas
        new_img = Image.new('RGB', target_size, (0, 0, 0))
        
        # Calculate centering offset
        offset = ((target_size[0] - img.width) // 2, (target_size[1] - img.height) // 2)
        
        new_img.paste(img, offset)
        new_img.save(output_path, 'JPEG', quality=85, optimize=True)

async def get_audio_duration(audio_path: Path) -> float:
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_path)]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, _ = await proc.communicate()
    try: 
        return float(stdout.decode().strip())
    except: 
        return 1.5

def cleanup_temp_dirs(*paths: Path):
    for p in paths:
        if p.exists(): shutil.rmtree(p, ignore_errors=True)
