import os
import asyncio
import logging
import subprocess
import aiohttp
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class TTSEngine:
    def __init__(self, engine_type: str = "edge_tts"):
        self.engine_type = engine_type
        self._models: Dict[str, Any] = {}
        # We'll use a shared session for all API calls
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))
        return self._session

    async def generate(self, text: str, output_path: str) -> bool:
        cleaned_text = self._clean_text(text)
        if not cleaned_text or len(cleaned_text) < 2:
            return await self._generate_silence(output_path)

        # Truncate to avoid API overhead/errors
        text_chunk = cleaned_text[:3000]
        
        # Strategy: Try Primary -> Try EdgeTTS -> Try Silence
        engines = [self.engine_type]
        if self.engine_type != "edge_tts":
            engines.append("edge_tts")

        for engine in engines:
            try:
                success = await self._call_engine(engine, text_chunk, output_path)
                if success and Path(output_path).exists() and Path(output_path).stat().st_size > 100:
                    return True
            except Exception as e:
                logger.error(f"TTS Engine {engine} failed: {e}")
                continue

        return await self._generate_silence(output_path)

    async def _call_engine(self, engine: str, text: str, output_path: str) -> bool:
        """Routes to the correct implementation."""
        if engine == "edge_tts":
            return await self._tts_edge(text, output_path)
        
        # API Based Engines
        session = await self._get_session()
        if engine == "elevenlabs":
            return await self._api_elevenlabs(session, text, output_path)
        if engine == "deepgram_aura":
            return await self._api_deepgram(session, text, output_path)
        
        # Local Heavy Engines (Run in ThreadPool to avoid blocking event loop)
        loop = asyncio.get_running_loop()
        if engine == "melo_tts":
            return await loop.run_in_executor(None, self._local_melo, text, output_path)
        if engine == "xtts_v2":
            return await loop.run_in_executor(None, self._local_xtts, text, output_path)
        
        return False

    # --- API Implementations (Non-blocking) ---

    async def _api_elevenlabs(self, session: aiohttp.ClientSession, text: str, output_path: str) -> bool:
        key = os.getenv("ELEVENLABS_API_KEY")
        voice_id = os.getenv("ELEVENLABS_VOICE_ID", "Xb7hH8MSUJpSbSDYk0k2")
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {"xi-api-key": key, "Content-Type": "application/json"}
        data = {"text": text, "model_id": "eleven_multilingual_v2"}

        async with session.post(url, json=data, headers=headers) as resp:
            if resp.status == 200:
                content = await resp.read()
                with open(output_path, "wb") as f:
                    f.write(content)
                return True
            return False

    async def _tts_edge(self, text: str, output_path: str) -> bool:
        from edge_tts import Communicate
        # Use a reliable neutral voice
        voice = os.getenv("EDGE_TTS_VOICE", "en-US-AndrewNeural")
        try:
            comm = Communicate(text, voice)
            await comm.save(output_path)
            return True
        except Exception:
            return False

    # --- Local Models (Memory Sensitive) ---

    def _local_xtts(self, text: str, output_path: str) -> bool:
        from TTS.api import TTS
        if "xtts" not in self._models:
            logger.info("Initializing XTTSv2 (High Memory Usage)...")
            self._models["xtts"] = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        
        self._models["xtts"].tts_to_file(text=text[:250], file_path=output_path, language="en")
        return True

    # --- Utility ---

    def _clean_text(self, text: str) -> str:
        """Aggressive cleaning for Manga OCR artifacts."""
        text = text.replace('\n', ' ')
        # Remove common OCR noise like | or [ ]
        import re
        text = re.sub(r'[|\[\]{}@]', '', text)
        return ' '.join(text.split()).strip()

    async def _generate_silence(self, output_path: str, duration: float = 1.0) -> bool:
        """Create silence using ffmpeg via async subprocess."""
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono",
            "-t", str(duration), "-q:a", "9", "-acodec", "libmp3lame", output_path
        ]
        proc = await asyncio.create_subprocess_exec(*cmd, stderr=asyncio.subprocess.PIPE)
        await proc.wait()
        return True

    async def cleanup(self):
        if self._session:
            await self._session.close()
        self._models.clear()
        import gc
        gc.collect()
        
