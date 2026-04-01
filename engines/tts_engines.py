import os
import asyncio
import requests
import torch
import wave
import numpy as np
from pathlib import Path
import time

class TTSEngine:
    def __init__(self, engine_type="edge_tts"):
        self.engine_type = engine_type
        self.model = None
        self.initialized_models = {}  # Support multiple model types
        print(f"Initialized TTS Engine: {self.engine_type}")

    async def generate(self, text, output_path):
        """Orchestrator for TTS generation with silent fallback logic."""
        # Validate input
        if not text or len(text.strip()) < 2:
            print(f"Text too short or empty, generating silence")
            return self._generate_silence(output_path)

        # Clean text for better TTS (remove excessive newlines, special chars)
        text = self._clean_text(text)

        try:
            if self.engine_type == "elevenlabs":
                return self._tts_elevenlabs(text, output_path)
            
            elif self.engine_type == "deepgram_aura":
                return self._tts_deepgram(text, output_path)
            
            elif self.engine_type == "fish_speech":
                return self._tts_fish_api(text, output_path)

            elif self.engine_type == "melo_tts":
                return self._tts_melo(text, output_path)

            elif self.engine_type == "chat_tts":
                return self._tts_chat(text, output_path)

            elif self.engine_type == "xtts_v2":
                return self._tts_xtts(text, output_path)

            else:
                return await self._tts_edge(text, output_path)

        except Exception as e:
            print(f"TTS Failure on {self.engine_type}: {e}. Falling back to Edge-TTS.")
            try:
                return await self._tts_edge(text, output_path)
            except Exception as fallback_error:
                print(f"Edge-TTS fallback also failed: {fallback_error}")
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
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    async def _tts_edge(self, text, output_path):
        """Free, reliable Microsoft Neural TTS."""
        try:
            from edge_tts import Communicate
            # Use a more natural voice with better pacing
            communicate = Communicate(text, "en-US-JennyNeural")
            await communicate.save(output_path)
            
            # Verify file was created
            if not Path(output_path).exists() or Path(output_path).stat().st_size == 0:
                raise Exception("Edge-TTS produced empty file")
            return True
        except ImportError:
            print("Edge-TTS not installed, generating silence")
            return self._generate_silence(output_path)

    def _tts_elevenlabs(self, text, output_path):
        """ElevenLabs API with better error handling."""
        key = os.getenv("ELEVENLABS_API_KEY")
        if not key:
            raise Exception("ELEVENLABS_API_KEY not set")
        
        # Use a more reliable voice (Adam) that's always available
        url = "https://api.elevenlabs.io/v1/text-to-speech/Xb7hH8MSUJpSbSDYk0k2"  # Adam voice
        headers = {"xi-api-key": key, "Content-Type": "application/json"}
        data = {
            "text": text[:5000],  # ElevenLabs has character limits
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
        data = {"text": text[:2000]}  # Deepgram character limit
        
        try:
            res = requests.post(url, json=data, headers=headers, timeout=30)
            if res.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(res.content)
                return True
            else:
                print(f"Deepgram returned {res.status_code}: {res.text}")
                return False
        except requests.RequestException as e:
            print(f"Deepgram request failed: {e}")
            return False

    def _tts_fish_api(self, text, output_path):
        """Fish Speech V1.5 API Integration."""
        key = os.getenv("FISH_KEY")
        if not key:
            raise Exception("FISH_KEY not set")
        
        url = "https://api.fish.audio/v1/tts"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        data = {
            "text": text[:1000],  # Fish API character limit
            "format": "mp3",
            "voice": "taylor"  # Default voice
        }
        
        try:
            res = requests.post(url, json=data, headers=headers, timeout=30)
            if res.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(res.content)
                return True
            else:
                print(f"Fish API returned {res.status_code}: {res.text}")
                return False
        except requests.RequestException as e:
            print(f"Fish API request failed: {e}")
            return False

    def _tts_melo(self, text, output_path):
        """MeloTTS - Optimized for CPU with better error handling."""
        try:
            from melotts.api import TTS
        except ImportError:
            print("MeloTTS not installed")
            raise
            
        try:
            if "melo" not in self.initialized_models:
                # Load model with CPU optimization
                self.initialized_models["melo"] = TTS(language='EN', device='cpu')
                print("MeloTTS model loaded")
            
            model = self.initialized_models["melo"]
            speaker_ids = model.hps.data.spk2id
            
            # Limit text length to avoid memory issues
            if len(text) > 500:
                text = text[:500]
                print(f"Truncated text to 500 chars for MeloTTS")
            
            # Generate speech
            model.tts_to_file(text, speaker_ids['EN-Default'], output_path, speed=0.9)
            
            # Verify output
            if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
                return True
            else:
                raise Exception("MeloTTS produced empty file")
                
        except Exception as e:
            print(f"MeloTTS error: {e}")
            raise

    def _tts_chat(self, text, output_path):
        """ChatTTS Implementation with memory optimization."""
        try:
            import ChatTTS
        except ImportError:
            print("ChatTTS not installed")
            raise
            
        try:
            if "chat" not in self.initialized_models:
                self.initialized_models["chat"] = ChatTTS.Chat()
                self.initialized_models["chat"].load_models(device='cpu')
                print("ChatTTS model loaded")
            
            model = self.initialized_models["chat"]
            
            # Limit text length
            if len(text) > 500:
                text = text[:500]
                print(f"Truncated text to 500 chars for ChatTTS")
            
            # Generate speech
            wavs = model.infer([text])
            
            # Save as WAV file
            import scipy.io.wavfile as wavfile
            wavfile.write(output_path, 24000, np.array(wavs[0]))
            
            # Convert to MP3 for consistency (optional)
            if output_path.endswith('.mp3'):
                mp3_path = output_path
                wav_path = output_path.replace('.mp3', '_temp.wav')
                Path(wav_path).write_bytes(Path(output_path).read_bytes())
                
                # Convert WAV to MP3 using ffmpeg
                import subprocess
                subprocess.run([
                    "ffmpeg", "-i", wav_path, "-c:a", "libmp3lame", 
                    "-q:a", "4", mp3_path, "-y"
                ], check=True, capture_output=True)
                Path(wav_path).unlink()
            
            return True
            
        except Exception as e:
            print(f"ChatTTS error: {e}")
            raise

    def _tts_xtts(self, text, output_path):
        """XTTS-v2 (Coqui) Implementation with CPU optimization."""
        try:
            from TTS.api import TTS
        except ImportError:
            print("TTS not installed")
            raise
            
        try:
            if "xtts" not in self.initialized_models:
                # Load lightweight XTTS model for CPU
                self.initialized_models["xtts"] = TTS(
                    "tts_models/multilingual/multi-dataset/xtts_v2"
                ).to("cpu")
                print("XTTS-v2 model loaded")
            
            model = self.initialized_models["xtts"]
            
            # Limit text length
            if len(text) > 500:
                text = text[:500]
                print(f"Truncated text to 500 chars for XTTS")
            
            # Use default speaker or create a reference file if needed
            speaker_wav = "scripts/ref.wav"
            if not Path(speaker_wav).exists():
                # Use a default voice without reference
                model.tts_to_file(
                    text=text, 
                    file_path=output_path, 
                    language="en"
                )
            else:
                model.tts_to_file(
                    text=text, 
                    file_path=output_path, 
                    speaker_wav=speaker_wav, 
                    language="en"
                )
            
            # Verify output
            if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
                return True
            else:
                raise Exception("XTTS produced empty file")
                
        except Exception as e:
            print(f"XTTS error: {e}")
            raise

    def _generate_silence(self, output_path, duration_seconds=1.0):
        """Zero-Fail fallback: creates a silent audio file."""
        try:
            # Generate silent MP3 using ffmpeg (more compatible)
            if output_path.endswith('.mp3'):
                import subprocess
                subprocess.run([
                    "ffmpeg", "-f", "lavfi", "-i", 
                    f"anullsrc=channel_layout=stereo:sample_rate=44100",
                    "-t", str(duration_seconds), "-c:a", "libmp3lame",
                    "-q:a", "9", output_path, "-y"
                ], check=True, capture_output=True)
            else:
                # Generate WAV silence
                with wave.open(output_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(44100)
                    frames = int(44100 * duration_seconds)
                    wf.writeframes(b'\x00' * frames * 2)
            
            print(f"Generated {duration_seconds}s silence file")
            return True
            
        except Exception as e:
            print(f"Silence generation failed: {e}")
            # Ultra-fallback: create empty file
            Path(output_path).touch()
            return False

    def cleanup(self):
        """Clean up loaded models to free memory."""
        self.initialized_models.clear()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
