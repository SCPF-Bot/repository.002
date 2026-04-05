import os
import cv2
import logging
import importlib.util
import re
from typing import List, Any
import google.generativeai as genai

# Global environment flags for Paddle optimization
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_allocator_strategy"] = "naive_best_fit"

logger = logging.getLogger(__name__)

class OCREngine:
    def __init__(self, engine_type: str = "tesseract", api_key: str = None):
        self.primary_engine = engine_type.lower()
        self.engines_to_try = [self.primary_engine]
        
        # Always fallback to Tesseract if the primary neural engine fails
        if self.primary_engine != "tesseract":
            self.engines_to_try.append("tesseract")
            
        self._model = None
        self._google_client = None

        # Initialize Gemini if API key is provided
        if api_key:
            logger.info("Initializing Gemini AI Text Cleaner...")
            genai.configure(api_key=api_key)
            self.ai_model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            logger.warning("No Google API Key found. AI cleaning will be skipped.")
            self.ai_model = None

    def _ai_clean_text(self, messy_text: str) -> str:
        """Uses AI to remove garbled OCR artifacts and fix grammar."""
        if not self.ai_model or len(messy_text) < 5:
            return messy_text

        prompt = (
            "You are a manga translation assistant. Clean the following OCR text from a speech bubble. "
            "Remove garbled characters, OCR artifacts, and nonsensical symbols. "
            "Fix the grammar to make it natural English. "
            "Only return the cleaned text, nothing else.\n\n"
            f"OCR Output: {messy_text}"
        )
        
        try:
            response = self.ai_model.generate_content(prompt)
            # Standard cleaning to remove AI markdown if it leaks through
            cleaned = response.text.strip().replace('*', '').replace('#', '')
            return cleaned
        except Exception as e:
            logger.warning(f"AI Cleaning failed: {e}")
            return messy_text

    def get_text(self, image_path: str) -> str:
        """
        Attempts OCR with the primary engine, then runs the result 
        through Gemini AI for grammar correction.
        """
        for engine in self.engines_to_try:
            try:
                method = getattr(self, f"_ocr_{engine}")
                text = method(image_path)
                
                if text and len(text.strip()) > 1:
                    # NEW: Apply AI cleanup before returning to core pipeline
                    return self._ai_clean_text(text.strip())
            except Exception as e:
                logger.error(f"Engine {engine} failed: {e}")
                continue
        return ""

    def _ocr_google_vision(self, image_path: str) -> str:
        if importlib.util.find_spec("google") is None:
            raise ImportError("google-cloud-vision not installed")
        from google.cloud import vision
        if self._google_client is None:
            self._google_client = vision.ImageAnnotatorClient()
        with open(image_path, "rb") as f:
            content = f.read()
        image = vision.Image(content=content)
        response = self._google_client.text_detection(image=image)
        if response.error.message:
            raise Exception(f"Google Vision API Error: {response.error.message}")
        return response.full_text_annotation.text if response.full_text_annotation else ""

    def _ocr_manga_ocr(self, image_path: str) -> str:
        from manga_ocr import MangaOCR
        if self._model is None:
            logger.info("Loading MangaOCR model...")
            self._model = MangaOCR()
        return self._model(image_path)

    def _ocr_paddle_ocr(self, image_path: str) -> str:
        from paddleocr import PaddleOCR
        import logging as py_logging
        py_logging.getLogger("ppocr").setLevel(py_logging.ERROR)
        if self._model is None:
            self._model = PaddleOCR(use_angle_cls=True, lang='en')
        result = self._model.ocr(image_path, cls=True)
        if not result or result[0] is None:
            return ""
        return " ".join([line[1][0] for line in result[0]])

    def _ocr_tesseract(self, image_path: str) -> str:
        import pytesseract
        img = cv2.imread(image_path)
        if img is None: return ""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        return pytesseract.image_to_string(processed, config='--psm 3')
