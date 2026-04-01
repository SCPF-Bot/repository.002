import os
import asyncio
import requests
import torch
import wave
import numpy as np
import subprocess
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TTSEngine:
    """TTS Engine with comprehensive failover and fallback mechanisms."""
    
    def __init__(self, engine_type="edge_tts"):
        self.engine_type = engine_type
        self.models = {}
        logger.info(f"Initialized TTS Engine: {self.engine_type}")

    async def generate(self, text, output_path):
        """Orchestrator for TTS generation with silent fallback logic."""
        # Validate input
        if not text or len(text.strip()) < 2:
            logger.info(f"Text too short or empty, generating silence")
            return self._generate_silence(output_path)

        # Clean text for better TTS
        text = self._clean_text(text)
        
        # Limit text length to prevent API issues
        if len(text) > 5000:
            text = text[:5000]
            logger.info(f"Truncated text to 5000 chars")

        # Try the selected engine with fallbacks
        engines_to_try = [self.engine_type, "edge_tts"]  # Always fallback to edge_tts
        
        for engine in engines_to_try:
            try:
                if engine == "elevenlabs":
                    success = self._tts_elevenlabs(text, output_path)
                elif engine == "deepgram_aura":
                    success = self._tts_deepgram(text, output_path)
                elif engine == "fish_speech":
                    success = self._tts_fish_api(text, output_path)
                elif engine == "melo_tts":
                    success = self._tts_melo(text, output_path)
                elif engine == "chat_tts":
                    success = self._tts_chat(text, output_path)
                elif engine == "xtts_v2":
                    success = self._tts_xtts(text, output_path)
                else:
                    success = await self._tts_edge(text, output_path)
                
                if success and Path(output_path).exists() and Path(output_path).stat().st_size > 0:
                    logger.info(f"TTS successful using {engine}, size: {Path(output_path).stat().st_size} bytes")
                    return True
                else:
                    logger.warning(f"TTS engine {engine} produced empty or failed output")
                    
            except Exception as e:
                logger.error(f"TTS engine {engine} failed: {e}")
                continue
        
        # Ultimate fallback - generate silence
        logger.error(f"All TTS engines failed for text: {text[:50]}...")
        return self._generate_silence(output_path)

    def _clean_text(self, text):
        """Clean text for better TTS pronunciation."""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Replace common problematic characters
        replacements = {
            '…': '...',
            '—': '-',
            '–': '-',
            '"': "'",
            '"': "'",
            '\u201c': '"',  # Left double quotation mark
            '\u201d': '"',  # Right double quotation mark
            '\u2018': "'",  # Left single quotation mark
            '\u2019': "'",  # Right single quotation mark
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove any remaining non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char == '\n')
        
        return text.strip()

    async def _tts_edge(self, text, output_path):
        """Free, reliable Microsoft Neural TTS."""
        try:
            from edge_tts import Communicate
            # Use a natural voice with good pacing
            communicate = Communicate(text, "en-US-JennyNeural")
            await communicate.save(output_path)
            
            # Verify file was created
            if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
                return True
            else:
                raise Exception("Edge-TTS produced empty file")
                
        except ImportError:
            logger.error("Edge-TTS not installed")
            raise
        except Exception as e:
            logger.error(f"Edge-TTS error: {e}")
            raise

    def _tts_elevenlabs(self, text, output_path):
        """ElevenLabs API with better error handling."""
        key = os.getenv("ELEVENLABS_API_KEY")
        if not key:
            raise Exception("ELEVENLABS_API_KEY not set")
        
        # Use a reliable voice that's always available
        url = "https://api.elevenlabs.io/v1/text-to-speech/Xb7hH8MSUJpSbSDYk0k2"  # Adam voice
        headers = {"xi-api-key": key, "Content-Type": "application/json"}
        data = {
            "text": text[:5000],
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
        }
        
        try:
            res = requests.post(url, json=data, headers=headers, timeout=30)
            if res.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(res.content)
                return True
            else:
                raise Exception(f"ElevenLabs Error {res.status_code}: {res.text}")
                
        except requests.RequestException as e:
            raise Exception(f"ElevenLabs request failed: {e}")

    def _tts_deepgram(self, text, output_path):
        """Deepgram Aura TTS."""
        key = os.getenv("DEEPGRAM_KEY")
        if not key:
            raise Exception("DEEPGRAM_KEY not set")
        
        url = "https://api.deepgram.com/v1/speak?model=aura-helios-en"
        headers = {"Authorization": f"Token {key}", "Content-Type": "application/json"}
        data = {"text": text[:2000]}
        
        try:
            res = requests.post(url, json=data, headers=headers, timeout=30)
            if res.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(res.content)
                return True
            else:
                raise Exception(f"Deepgram returned {res.status_code}: {res.text}")
                
        except requests.RequestException as e:
            raise Exception(f"Deepgram request failed: {e}")

    def _tts_fish_api(self, text, output_path):
        """Fish Speech V1.5 API Integration."""
        key = os.getenv("FISH_KEY")
        if not key:
            raise Exception("FISH_KEY not set")
        
        url = "https://api.fish.audio/v1/tts"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        data = {
            "text": text[:1000],
            "format": "mp3",
            "voice": "taylor"
        }
        
        try:
            res = requests.post(url, json=data, headers=headers, timeout=30)
            if res.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(res.content)
                return True
            else:
                raise Exception(f"Fish API returned {res.status_code}: {res.text}")
                
        except requests.RequestException as e:
            raise Exception(f"Fish API request failed: {e}")

    def _tts_melo(self, text, output_path):
        """MeloTTS - Optimized for CPU."""
        try:
            from melotts.api import TTS
        except ImportError:
            logger.error("MeloTTS not installed")
            raise
        
        try:
            model_key = "melo"
            if model_key not in self.models:
                self.models[model_key] = TTS(language='EN', device='cpu')
                logger.info("MeloTTS model loaded")
            
            model = self.models[model_key]
            speaker_ids = model.hps.data.spk2id
            
            # Limit text length
            if len(text) > 500:
                text = text[:500]
                logger.info(f"Truncated text to 500 chars for MeloTTS")
            
            # Generate speech
            model.tts_to_file(text, speaker_ids['EN-Default'], output_path, speed=0.9)
            
            # Verify output
            if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
                return True
            else:
                raise Exception("MeloTTS produced empty file")
                
        except Exception as e:
            logger.error(f"MeloTTS error: {e}")
            raise

    def _tts_chat(self, text, output_path):
        """ChatTTS Implementation with memory optimization."""
        try:
            import ChatTTS
        except ImportError:
            logger.error("ChatTTS not installed")
            raise
        
        try:
            model_key = "chat"
            if model_key not in self.models:
                self.models[model_key] = ChatTTS.Chat()
                self.models[model_key].load_models(device='cpu')
                logger.info("ChatTTS model loaded")
            
            model = self.models[model_key]
            
            # Limit text length
            if len(text) > 500:
                text = text[:500]
                logger.info(f"Truncated text to 500 chars for ChatTTS")
            
            # Generate speech
            wavs = model.infer([text])
            
            # Save as WAV file
            temp_wav = output_path.replace('.mp3', '_temp.wav')
            import scipy.io.wavfile as wavfile
            wavfile.write(temp_wav, 24000, np.array(wavs[0]))
            
            # Convert to MP3
            subprocess.run([
                "ffmpeg", "-i", temp_wav, "-c:a", "libmp3lame", 
                "-q:a", "4", output_path, "-y"
            ], check=True, capture_output=True)
            
            # Clean up temp file
            Path(temp_wav).unlink()
            
            return True
            
        except Exception as e:
            logger.error(f"ChatTTS error: {e}")
            raise

    def _tts_xtts(self, text, output_path):
        """XTTS-v2 (Coqui) Implementation with CPU optimization."""
        try:
            from TTS.api import TTS
        except ImportError:
            logger.error("TTS not installed")
            raise
        
        try:
            model_key = "xtts"
            if model_key not in self.models:
                self.models[model_key] = TTS(
                    "tts_models/multilingual/multi-dataset/xtts_v2"
                ).to("cpu")
                logger.info("XTTS-v2 model loaded")
            
            model = self.models[model_key]
            
            # Limit text length
            if len(text) > 500:
                text = text[:500]
                logger.info(f"Truncated text to 500 chars for XTTS")
            
            # Use default voice without reference file
            model.tts_to_file(
                text=text, 
                file_path=output_path, 
                language="en"
            )
            
            # Verify output
            if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
                return True
            else:
                raise Exception("XTTS produced empty file")
                
        except Exception as e:
            logger.error(f"XTTS error: {e}")
            raise

    def _generate_silence(self, output_path, duration_seconds=1.5):
        """Zero-Fail fallback: creates a silent audio file."""
        try:
            # Generate silent MP3 using ffmpeg
            subprocess.run([
                "ffmpeg", "-f", "lavfi", "-i", 
                f"anullsrc=channel_layout=stereo:sample_rate=44100",
                "-t", str(duration_seconds), "-c:a", "libmp3lame",
                "-q:a", "9", output_path, "-y"
            ], check=True, capture_output=True, timeout=10)
            
            logger.info(f"Generated {duration_seconds}s silence file")
            return True
            
        except Exception as e:
            logger.error(f"Silence generation failed: {e}")
            # Ultra-fallback: create empty file
            try:
                Path(output_path).touch()
            except:
                pass
            return False

    def cleanup(self):
        """Clean up loaded models to free memory."""
        self.models.clear()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
