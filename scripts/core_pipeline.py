#!/usr/bin/env python3
"""
Manga to Video AI Pipeline
Orchestrates OCR, TTS, and video rendering for manga chapters.
"""

import os
import sys
import argparse
import logging
import zipfile
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
import json
import time

# Add engines directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engines.ocr_engines import OCREngine
from engines.tts_engines import TTSEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MangaToVideoPipeline:
    """Main pipeline orchestrator for manga to video conversion."""
    
    def __init__(self, ocr_engine="tesseract", tts_engine="edge_tts", debug=False):
        self.ocr_engine = OCREngine(ocr_engine)
        self.tts_engine = TTSEngine(tts_engine)
        self.debug = debug
        
        # Setup directories
        self.base_dir = Path("processing")
        self.images_dir = self.base_dir / "images"
        self.audio_dir = self.base_dir / "audio"
        self.output_dir = Path("output")
        
        # Create directories
        for d in [self.base_dir, self.images_dir, self.audio_dir, self.output_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Pipeline initialized: OCR={ocr_engine}, TTS={tts_engine}")
    
    def download_and_extract(self, url: str) -> List[Path]:
        """Download and extract manga archive."""
        logger.info(f"Processing URL: {url}")
        
        # Handle different URL types
        if url.startswith(('http://', 'https://')):
            # Download file
            import requests
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            temp_zip = self.base_dir / "temp_manga.zip"
            with open(temp_zip, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            archive_path = temp_zip
        else:
            # Local file
            archive_path = Path(url)
        
        # Extract images
        if archive_path.suffix.lower() in ['.cbz', '.zip']:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(self.images_dir)
        else:
            # Assume it's a directory or single image
            if archive_path.is_file():
                shutil.copy(archive_path, self.images_dir)
            elif archive_path.is_dir():
                shutil.copytree(archive_path, self.images_dir, dirs_exist_ok=True)
        
        # Get sorted image files
        image_files = self._get_sorted_images()
        logger.info(f"Extracted {len(image_files)} images")
        
        # Cleanup temp file
        if 'temp_zip' in locals():
            temp_zip.unlink()
        
        return image_files
    
    def _get_sorted_images(self) -> List[Path]:
        """Get sorted list of image files."""
        from natsort import natsorted
        
        extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        images = []
        
        for ext in extensions:
            images.extend(self.images_dir.glob(f"*{ext}"))
            images.extend(self.images_dir.glob(f"*{ext.upper()}"))
        
        return natsorted(images)
    
    def extract_text(self, image_path: Path) -> str:
        """Extract text from image using OCR."""
        logger.info(f"Extracting text from: {image_path.name}")
        return self.ocr_engine.get_text(str(image_path))
    
    def generate_audio(self, text: str, output_path: Path) -> bool:
        """Generate audio from text using TTS."""
        if not text or len(text.strip()) < 2:
            logger.warning(f"Empty text for {output_path.name}, generating silence")
            return self.tts_engine._generate_silence(str(output_path))
        
        logger.info(f"Generating audio for: {output_path.name} ({len(text)} chars)")
        
        # Use asyncio to run async TTS
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.tts_engine.generate(text, str(output_path))
            )
            loop.close()
            return result
        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            return False
    
    def process_pages(self, image_files: List[Path]) -> List[Dict]:
        """Process all pages through OCR and TTS."""
        results = []
        
        for idx, img_path in enumerate(image_files):
            logger.info(f"Processing page {idx + 1}/{len(image_files)}")
            
            # Extract text
            text = self.extract_text(img_path)
            
            if self.debug:
                logger.debug(f"Page {idx + 1} text: {text[:200]}...")
            
            # Generate audio
            audio_path = self.audio_dir / f"page_{idx + 1:04d}.mp3"
            success = self.generate_audio(text, audio_path)
            
            results.append({
                'page': idx + 1,
                'image': str(img_path),
                'audio': str(audio_path) if success else None,
                'text': text,
                'has_audio': success
            })
            
            # Save text for debugging
            if self.debug:
                text_path = self.base_dir / f"page_{idx + 1:04d}.txt"
                text_path.write_text(text, encoding='utf-8')
        
        return results
    
    def render_video(self, results: List[Dict]) -> Path:
        """Render final video with images and audio."""
        logger.info("Rendering final video...")
        
        # Create a concat file for ffmpeg
        concat_file = self.base_dir / "concat.txt"
        audio_sources = []
        
        with open(concat_file, 'w') as f:
            for result in results:
                img_path = result['image']
                audio_path = result.get('audio')
                
                if audio_path and Path(audio_path).exists():
                    # Calculate duration based on audio length
                    duration = self._get_audio_duration(audio_path)
                    audio_sources.append(audio_path)
                else:
                    # Default duration for silent pages
                    duration = 3.0
                
                f.write(f"file '{img_path}'\n")
                f.write(f"duration {duration}\n")
        
        # Generate video with audio
        output_video = self.output_dir / "manga_video.mp4"
        
        if audio_sources:
            # Mix all audio tracks
            mixed_audio = self._mix_audio_files(audio_sources)
            audio_input = ["-i", str(mixed_audio)]
        else:
            audio_input = []
        
        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            *audio_input,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            "-y",
            str(output_video)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                raise Exception("Video rendering failed")
            
            logger.info(f"Video rendered: {output_video}")
            return output_video
            
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg timed out")
            raise
        except Exception as e:
            logger.error(f"FFmpeg failed: {e}")
            raise
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio file duration in seconds."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout:
                return float(result.stdout.strip())
        except:
            pass
        
        return 3.0  # Default duration
    
    def _mix_audio_files(self, audio_files: List[str]) -> Path:
        """Mix multiple audio files into a single track."""
        mixed_audio = self.base_dir / "mixed_audio.mp3"
        
        # Create concat for audio
        audio_concat = self.base_dir / "audio_concat.txt"
        with open(audio_concat, 'w') as f:
            for audio in audio_files:
                f.write(f"file '{audio}'\n")
        
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", str(audio_concat),
            "-c:a", "libmp3lame",
            "-b:a", "128k",
            "-y",
            str(mixed_audio)
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        return mixed_audio
    
    def cleanup(self):
        """Clean up temporary files."""
        if not self.debug:
            logger.info("Cleaning up temporary files...")
            shutil.rmtree(self.base_dir, ignore_errors=True)
        else:
            logger.info(f"Debug mode: keeping files in {self.base_dir}")
        
        # Cleanup loaded models
        self.ocr_engine.cleanup()
        self.tts_engine.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Manga to Video AI Pipeline")
    parser.add_argument("--url", required=True, help="Manga download URL or local path")
    parser.add_argument("--ocr", default="tesseract", 
                       choices=["google_vision", "manga_ocr", "paddle_ocr", 
                               "comic_text_detector", "tesseract"],
                       help="Primary OCR engine")
    parser.add_argument("--tts", default="edge_tts",
                       choices=["elevenlabs", "fish_speech", "xtts_v2", 
                               "chat_tts", "melo_tts", "deepgram_aura", "edge_tts"],
                       help="Primary TTS engine")
    parser.add_argument("--enable-debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.enable_debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    pipeline = None
    try:
        pipeline = MangaToVideoPipeline(
            ocr_engine=args.ocr,
            tts_engine=args.tts,
            debug=args.enable_debug
        )
        
        # Download and extract
        image_files = pipeline.download_and_extract(args.url)
        
        if not image_files:
            logger.error("No images found in the archive")
            sys.exit(1)
        
        # Process pages
        results = pipeline.process_pages(image_files)
        
        # Count successful pages
        successful = sum(1 for r in results if r['has_audio'])
        logger.info(f"Processed {len(results)} pages, {successful} with audio")
        
        # Render video
        video_path = pipeline.render_video(results)
        
        logger.info(f"✓ Pipeline completed successfully!")
        logger.info(f"Output video: {video_path}")
        logger.info(f"File size: {video_path.stat().st_size / (1024*1024):.1f} MB")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if pipeline:
            pipeline.cleanup()


if __name__ == "__main__":
    main()
