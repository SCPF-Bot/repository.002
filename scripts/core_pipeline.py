#!/usr/bin/env python3
"""
Manga AI Video Pipeline - Zero Fail Promise
Orchestrates the complete workflow from manga download to video rendering.
"""

import os
import argparse
import asyncio
import subprocess
import shutil
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import engines
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fallback for natsort
try:
    from natsort import natsorted
except ImportError:
    logger.warning("natsort not installed, using basic sort")
    natsorted = sorted

# Import engines
try:
    from engines.ocr_engines import OCREngine
    from engines.tts_engines import TTSEngine
except ImportError as e:
    logger.error(f"Failed to import engines: {e}")
    logger.error("Make sure engines/ directory exists with __init__.py, ocr_engines.py, and tts_engines.py")
    sys.exit(1)

# Global Configuration
BASE_DIR = Path("processing")
OUT_DIR = Path("output")
TEMP_IMG = BASE_DIR / "images"
TEMP_AUD = BASE_DIR / "audio"


class MangaVideoOrchestrator:
    """Main orchestrator for manga to video conversion."""
    
    def __init__(self, ocr_type, tts_type):
        logger.info(f"Initializing Orchestrator with OCR={ocr_type}, TTS={tts_type}")
        self.ocr = OCREngine(ocr_type)
        self.tts = TTSEngine(tts_type)
        self._prepare_env()

    def _prepare_env(self):
        """Clean and recreate the workspace for a fresh build."""
        try:
            for d in [OUT_DIR, TEMP_IMG, TEMP_AUD, BASE_DIR]:
                if d.exists():
                    shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)
            logger.info("Environment prepared successfully")
        except Exception as e:
            logger.error(f"Failed to prepare environment: {e}")
            raise

    async def download_and_extract(self, url):
        """Secure ingestion of manga assets."""
        logger.info("--- Phase 1: Ingestion ---")
        archive_path = BASE_DIR / "manga_archive.zip"
        
        try:
            # Download with curl (more reliable than wget in CI environments)
            logger.info(f"Downloading from: {url}")
            result = subprocess.run(
                ["curl", "-L", "--max-time", "300", "--fail", url, "-o", str(archive_path)], 
                check=True, 
                timeout=300,
                capture_output=True
            )
            logger.info(f"Download complete: {archive_path.stat().st_size} bytes")
            
            # Determine archive type and extract accordingly
            if archive_path.suffix.lower() == '.zip':
                subprocess.run(
                    ["unzip", "-o", "-q", str(archive_path), "-d", str(TEMP_IMG)], 
                    check=True, 
                    timeout=120,
                    capture_output=True
                )
            else:
                # Handle CBZ (which is just zip with different extension)
                subprocess.run(
                    ["unzip", "-o", "-q", str(archive_path), "-d", str(TEMP_IMG)], 
                    check=True, 
                    timeout=120,
                    capture_output=True
                )
            
            logger.info(f"Successfully extracted assets to {TEMP_IMG}")
            
            # Remove zip file to save space
            archive_path.unlink()
            
            # Handle nested directories - move files up if needed
            self._flatten_extracted_files()
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Download/Unzip failed: {e}")
            if e.stderr:
                logger.error(f"Error output: {e.stderr.decode()}")
            raise
        except subprocess.TimeoutExpired as e:
            logger.error(f"Operation timed out: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during ingestion: {e}")
            raise

    def _flatten_extracted_files(self):
        """Move images from nested directories to the root images directory."""
        try:
            # Find all image files in subdirectories
            valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif')
            for root, dirs, files in os.walk(TEMP_IMG):
                for file in files:
                    if file.lower().endswith(valid_exts):
                        src = Path(root) / file
                        dst = TEMP_IMG / file
                        if src != dst:
                            shutil.move(str(src), str(dst))
            
            # Remove empty directories
            for root, dirs, files in os.walk(TEMP_IMG, topdown=False):
                for dir_name in dirs:
                    try:
                        dir_path = Path(root) / dir_name
                        if dir_path.exists() and not any(dir_path.iterdir()):
                            dir_path.rmdir()
                    except Exception:
                        pass
                        
            logger.info("Extracted files flattened")
            
        except Exception as e:
            logger.warning(f"Error flattening files: {e}")

    async def run(self, url):
        """Main execution pipeline."""
        try:
            # Phase 1: Download and extract
            await self.download_and_extract(url)

            # Phase 2: Process pages
            await self._process_pages()

            # Phase 3: Render video
            await self._render_video()
            
            logger.info(f"Pipeline complete! Output in: {OUT_DIR}")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            self.cleanup()

    async def _process_pages(self):
        """Process all pages with OCR and TTS."""
        logger.info("--- Phase 2: Processing Pages ---")
        
        # Gather images and sort them naturally
        valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
        images = natsorted([
            f for f in TEMP_IMG.rglob("*") 
            if f.suffix.lower() in valid_exts and not f.name.startswith('.')
        ])

        if not images:
            raise Exception("No valid images found in the archive")

        logger.info(f"Found {len(images)} images to process")
        
        self.manifest_data = []
        self.audio_list_data = []
        self.page_durations = []

        for i, img_path in enumerate(images):
            logger.info(f"Processing Page {i+1}/{len(images)}: {img_path.name}")
            
            # 1. OCR Extraction
            try:
                text = self.ocr.get_text(str(img_path))
                if not text or not text.strip():
                    text = "No text detected on this page."
                    logger.warning(f"No text detected on page {i+1}")
                else:
                    logger.info(f"Extracted text: {text[:100]}...")
            except Exception as e:
                logger.error(f"OCR Error on page {i+1}: {e}")
                text = "Error processing text on this page."
            
            # 2. TTS Generation
            audio_file = TEMP_AUD / f"page_{i:04d}.mp3"
            try:
                await self.tts.generate(text, str(audio_file))
                if not audio_file.exists() or audio_file.stat().st_size == 0:
                    raise Exception(f"Audio file not created or empty: {audio_file}")
                logger.info(f"Generated audio: {audio_file.name} ({audio_file.stat().st_size} bytes)")
            except Exception as e:
                logger.error(f"TTS Error on page {i+1}: {e}")
                self._generate_silence_fallback(audio_file)
                logger.info(f"Created silent fallback audio")

            # 3. Get Duration
            duration = self._get_audio_duration(audio_file)
            self.page_durations.append(duration)

            # 4. Prepare FFmpeg Concatenation Strings
            safe_img_path = str(img_path.resolve()).replace("'", "'\\''")
            safe_aud_path = str(audio_file.resolve()).replace("'", "'\\''")
            
            self.manifest_data.append(f"file '{safe_img_path}'\nduration {duration}")
            self.audio_list_data.append(f"file '{safe_aud_path}'")

    def _get_audio_duration(self, audio_file):
        """Get audio duration using ffprobe."""
        try:
            result = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(audio_file)
            ], capture_output=True, timeout=10, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                duration = float(result.stdout.strip())
                # Cap duration to reasonable limits
                return min(max(duration, 1.0), 30.0)
            else:
                return 2.0
                
        except Exception as e:
            logger.warning(f"Failed to get duration for {audio_file}: {e}")
            return 2.0

    def _generate_silence_fallback(self, output_path, duration_seconds=2.0):
        """Generate silent audio file as fallback."""
        try:
            subprocess.run([
                "ffmpeg", "-f", "lavfi", "-i", 
                f"anullsrc=channel_layout=stereo:sample_rate=44100",
                "-t", str(duration_seconds), "-c:a", "libmp3lame",
                "-q:a", "9", str(output_path), "-y"
            ], check=True, capture_output=True, timeout=30)
        except Exception as e:
            logger.error(f"Failed to generate silence: {e}")
            # Ultra-fallback: create empty file
            try:
                output_path.touch()
            except:
                pass

    async def _render_video(self):
        """Render final video from processed pages."""
        logger.info("--- Phase 3: Final Encoding ---")
        
        if not self.manifest_data or not self.audio_list_data:
            raise Exception("No pages processed for rendering")
        
        # Write temporary manifests
        img_manifest = BASE_DIR / "img_list.txt"
        aud_manifest = BASE_DIR / "aud_list.txt"
        
        # FFmpeg concat demuxer requires the last file to be repeated without duration
        last_img = self.manifest_data[-1].split('\n')[0]
        img_manifest.write_text("\n".join(self.manifest_data) + f"\n{last_img}")
        aud_manifest.write_text("\n".join(self.audio_list_data))

        master_audio = BASE_DIR / "master_audio.mp3"
        final_video = OUT_DIR / "manga_ai_render.mp4"

        # 1. Combine all audio segments
        try:
            subprocess.run([
                "ffmpeg", "-f", "concat", "-safe", "0", "-i", str(aud_manifest),
                "-c", "copy", str(master_audio), "-y"
            ], check=True, capture_output=True, timeout=300)
            logger.info(f"Combined {len(self.audio_list_data)} audio tracks")
        except subprocess.CalledProcessError as e:
            logger.error(f"Audio concatenation failed: {e.stderr.decode() if e.stderr else str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during audio concatenation: {e}")
            raise

        # 2. Stitch images with audio
        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0", "-i", str(img_manifest),
            "-i", str(master_audio),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k", "-shortest", str(final_video), "-y"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            if final_video.exists():
                file_size = final_video.stat().st_size / (1024 * 1024)
                logger.info(f"BUILD SUCCESSFUL: {final_video} ({file_size:.2f} MB)")
            else:
                raise Exception("Output video file not created")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Video render failed: {e.stderr.decode() if e.stderr else str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during video render: {e}")
            raise

    def cleanup(self):
        """Clean up large intermediate files to save space."""
        try:
            for d in [TEMP_IMG, TEMP_AUD]:
                if d.exists():
                    shutil.rmtree(d)
            # Also clean up manifest files
            for f in BASE_DIR.glob("*.txt"):
                f.unlink()
            logger.info("Cleanup complete")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
        
        # Clean up engine resources
        if hasattr(self, 'ocr'):
            self.ocr.cleanup()
        if hasattr(self, 'tts'):
            self.tts.cleanup()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Zero-Fail Manga Pipeline")
    parser.add_argument("--url", required=True, help="URL to CBZ/Zip file")
    parser.add_argument("--ocr", default="tesseract", help="OCR Engine choice")
    parser.add_argument("--tts", default="edge_tts", help="TTS Engine choice")
    args = parser.parse_args()

    # Validate inputs
    if not args.url:
        logger.error("URL is required")
        sys.exit(1)

    orchestrator = None
    try:
        orchestrator = MangaVideoOrchestrator(args.ocr, args.tts)
        await orchestrator.run(args.url)
        logger.info("Pipeline completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
        
    finally:
        if orchestrator:
            orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
