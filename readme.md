# 🌀 Manga-to-Video

***This repository is a high-performance orchestration layer that turns static manga pages into narrated videos. It’s built to be stateless, serverless, and resilient.***

---

## 🛑 0. Prerequisites
Before you even think about clicking "Run," you need:
 * A GitHub Account: Obviously.
 * API Keys: While edge_tts and tesseract are free, they are "budget" options. For quality, get keys for:
   * ElevenLabs (Premium Voice)
   * Deepgram (Fastest Voice)
   * Google Cloud Vision (Best OCR)
 * A Direct Download URL: The pipeline needs a link to a .zip or .cbz file. It won't scrape a website; it expects a raw archive.

## 🛠 1. The Setup
### Step 1: Forking & Initialization
 * Fork this repository to your account.
 * Navigate to the Actions tab.
 * Click the big green button: "I understand my workflows, let them run."
### Step 2: Permissions
By default, GitHub Actions are "Read Only." This pipeline needs to upload video files to your "Releases."
 * Go to Settings > Actions > General.
 * Scroll to Workflow permissions.
 * Select Read and write permissions.
 * Click Save.
### Step 3: Hardening with Secrets
The code is built to handle API failures, but it can't invent keys you didn't provide.
 * Go to Settings > Secrets and variables > Actions.
 * Create New repository secret for any of the following you plan to use:
   * ELEVENLABS_API_KEY
   * DEEPGRAM_KEY
   * FISH_KEY
   * GOOGLE_CREDENTIALS (Paste the entire JSON service account key here).

## 🚀 2. Running the Pipeline
 * Go to the Actions tab.
 * Select the Manga-To-Video (High Performance) workflow on the left.
 * Click the Run workflow dropdown.
 * Inputs:
   * URL: Provide the direct link to your .zip or .cbz.
   * OCR Engine: Select google_vision for accuracy or tesseract for free.
   * TTS Engine: Select elevenlabs for quality or edge_tts for free.
 * Click Run workflow.

## 🏗 3. How/Why This Works
The architecture follows a strict Strategy Pattern to ensure the pipeline doesn't crash just because one API is having a bad day.
The "Zero-Fail" Logic:
 * Standardization: Every image is resized to 1920x1080 and padded with black bars before FFmpeg touches it. This prevents "Aspect Ratio Mismatch" errors that plague amateur scripts.
 * Concat Demuxing: We don't render 50 individual mini-videos. We generate 50 audio files, one image list, and tell FFmpeg to "stream-copy" them together. This is 10x faster than traditional rendering.
 * Exponential Backoff: If the TTS API returns a 429 Too Many Requests, the script doesn't quit. It waits 2^n seconds and tries again.
 * Memory Management: We run gc.collect() and clear CUDA caches after every page. GitHub's 7GB RAM limit is tight; we respect it.

## 📉 4. Troubleshooting

| Issue | Likely Cause | Solution |
|---|---|---|
| Workflow fails immediately | Permissions | See Step 2 above. You forgot to enable "Write" permissions. |
| No Text Found | Poor Image Quality | Tesseract is "okay," but it struggles with vertical text or stylized fonts. Use manga_ocr or google_vision. |
| Audio is cut off | FFmpeg Quirk | The new pipeline uses ffprobe to get exact durations. If it's still cutting off, check if your archive has corrupted images. |
| Pipeline is slow | Local Engines | xtts_v2 and manga_ocr are massive models. On GitHub's CPU-only runners, they take time. Use API-based engines for speed. |

## 💻 5. Local Development
If you want to run this on your own 4090-equipped rig:
 * Clone: git clone https://github.com/your-username/manga-to-video
 * System Deps: Run ./scripts/system_deps.sh.
 * Install: pip install -r requirements.txt.
 * Run:
   python scripts/core_pipeline.py --url "http://example.com/manga.zip" --ocr "manga_ocr" --tts "edge_tts"

---

# Disclaimer

***This tool is for automation. Please respect copyright laws and the Terms of Service of the API providers you use. Avoid using this for malicious acts such as pirating or any copyright infringement violations. If you hit a rate limit, don't blame the code please...***
