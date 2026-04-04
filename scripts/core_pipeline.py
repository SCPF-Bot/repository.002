#!/usr/bin/env python3
import os
import sys
import argparse
import asyncio
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ocr_engines import OCREngine
from tts_engines import TTSEngine
from utils import download_file, extract_archive, resize_and_pad, get_audio_duration, cleanup_temp_dirs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MangaPipeline")

class MangaToVideoPipeline:
    def __init__(self, url: str, ocr_engine: str, tts_engine: str):
        self.url = url
        self.ocr = OCREngine(ocr_engine)
        self.tts = TTSEngine(tts_engine)
        
        # ANCHOR TO REPO ROOT
        self.repo_root = SCRIPT_DIR.parent
        self.output_dir = self.repo_root / "output"
        self.output_dir.mkdir(exist_ok=True)
        self.output_video = self.output_dir / "final_manga_video.mp4"
        
        self.temp_dir = Path(tempfile.mkdtemp(prefix="manga_job_"))
        self.dirs = {
            "images": self.temp_dir / "images",
            "processed": self.temp_dir / "processed",
            "audio": self.temp_dir / "audio"
        }
        for d in self.dirs.values(): d.mkdir(parents=True, exist_ok=True)

    async def process_page(self, idx: int, orig_img: Path) -> Tuple[Path, Path, float]:
        proc_img = self.dirs["processed"] / f"page_{idx:04d}.jpg"
        await asyncio.to_thread(resize_and_pad, orig_img, proc_img)

        text = await asyncio.to_thread(self.ocr.get_text, str(proc_img))
        text = text.strip() if text.strip() else "..."

        audio_file = self.dirs["audio"] / f"audio_{idx:04d}.mp3"
        await self.tts.generate(text, str(audio_file))
        
        duration = await asyncio.to_thread(get_audio_duration, audio_file)
        return proc_img, audio_file, duration

    async def run(self) -> Path:
        try:
            archive_path = self.temp_dir / "manga.archive"
            await download_file(self.url, archive_path)
            image_paths = await extract_archive(archive_path, self.dirs["images"])
            
            semaphore = asyncio.Semaphore(4) 
            async def sem_task(i, path):
                async with semaphore: return await self.process_page(i, path)

            tasks = [sem_task(i, path) for i, path in enumerate(image_paths)]
            segments = await asyncio.gather(*tasks)

            return await self._render_final_video(segments)
        finally:
            self.tts.cleanup()
            cleanup_temp_dirs(self.temp_dir)

    async def _render_final_video(self, segments: List[Tuple[Path, Path, float]]) -> Path:
        logger.info("🎬 Rendering final video...")
        
        concat_meta = self.temp_dir / "meta.txt"
        with open(concat_meta, "w") as f:
            for img, audio, dur in segments:
                f.write(f"file '{img.absolute()}'\nduration {dur}\n")
            f.write(f"file '{segments[-1][0].absolute()}'\n")

        audio_concat = self.temp_dir / "audio_list.txt"
        with open(audio_concat, "w") as f:
            for _, audio, _ in segments:
                f.write(f"file '{audio.absolute()}'\n")

        final_audio = self.temp_dir / "final_audio.mp3"
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(audio_concat), "-c", "copy", str(final_audio)], check=True)

        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_meta),
            "-i", str(final_audio), "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-crf", "23", "-c:a", "aac", "-shortest",
            str(self.output_video.absolute())
        ]
        
        await asyncio.to_thread(subprocess.run, cmd, check=True)
        logger.info(f"✅ Video created: {self.output_video}")
        return self.output_video

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--ocr", default="tesseract")
    parser.add_argument("--tts", default="edge_tts")
    args = parser.parse_args()

    pipeline = MangaToVideoPipeline(args.url, args.ocr, args.tts)
    output = await pipeline.run()
    print(f"SUCCESS: {output}")

if __name__ == "__main__":
    asyncio.run(main())
