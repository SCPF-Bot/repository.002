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
        self._models = {}
        self._session = None

    async def _get_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))
        return self._session

    async def generate(self, text: str, output_path: str) -> bool:
        cleaned_text = self._clean_text(text)[:3000]
        # Handle cases where OCR returns nothing or just noise
        if not cleaned_text or len(cleaned_text.strip()) < 2: 
            return await self._generate_silence(output_path)
        
        engines = [self.engine_type]
        if self.engine_type != "edge_tts": 
            engines.append("edge_tts")

        for engine in engines:
            try:
                if await self._call_engine(engine, cleaned_text, output_path): 
                    return True
            except Exception as e:
                logger.warning(f"Engine {engine} failed: {e}")
                continue
        return await self._generate_silence(output_path)

    async def _call_engine(self, engine, text, output_path):
        if engine == "edge_tts": 
            return await self._tts_edge(text, output_path)
        
        session = await self._get_session()
        if engine == "elevenlabs": 
            return await self._api_elevenlabs(session, text, output_path)
        
        loop = asyncio.get_running_loop()
        if engine == "melo_tts": 
            return await loop.run_in_executor(None, self._local_melo, text, output_path)
        if engine == "xtts_v2": 
            return await loop.run_in_executor(None, self._local_xtts, text, output_path)
        return False

    async def _api_elevenlabs(self, session, text, output_path):
        key = os.getenv("ELEVENLABS_API_KEY")
        voice = os.getenv("ELEVENLABS_VOICE_ID", "Xb7hH8MSUJpSbSDYk0k2")
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
        
        if not key:
            return False

        async with session.post(url, json={"text": text, "model_id": "eleven_multilingual_v2"}, headers={"xi-api-key": key}) as resp:
            if resp.status == 200:
                with open(output_path, "wb") as f: 
                    f.write(await resp.read())
                return True
        return False

    async def _tts_edge(self, text, output_path):
        from edge_tts import Communicate
        voice = os.getenv("EDGE_TTS_VOICE", "en-US-AndrewNeural")
        try:
            await Communicate(text, voice).save(output_path)
            return True
        except: 
            return False

    def _local_xtts(self, text, output_path):
        from TTS.api import TTS
        if "xtts" not in self._models: 
            self._models["xtts"] = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        self._models["xtts"].tts_to_file(text=text[:250], file_path=output_path, language="en")
        return True

    def _clean_text(self, text):
        import re
        # Removes common OCR hallucinations and newlines
        return re.sub(r'[|\[\]{}@]', '', text.replace('\n', ' ')).strip()

    async def _generate_silence(self, output_path, duration=1.0):
        # Generates a clean silent track if OCR/TTS fails
        cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", str(duration), "-acodec", "libmp3lame", output_path]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        await proc.wait()
        return True

    async def cleanup(self):
        # FIXED: Correctly defined as async and awaits the session close
        if self._session and not self._session.closed:
            await self._session.close()
        self._models.clear()
        import gc
        gc.collect()
