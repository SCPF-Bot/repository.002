import os
import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class OCREngine:
    def __init__(self, engine_type="tesseract"):
        self.engine_type = engine_type
        self.model = None

    def get_text(self, image_path):
        # Fallback logic remains, but imports are scoped locally
        try:
            if self.engine_type == "google_vision":
                return self._ocr_google_vision(image_path)
            elif self.engine_type == "manga_ocr":
                return self._ocr_manga_ocr(image_path)
            else:
                return self._ocr_tesseract(image_path)
        except Exception as e:
            logger.error(f"Selected engine {self.engine_type} failed: {e}. Falling back to Tesseract.")
            return self._ocr_tesseract(image_path)

    def _ocr_manga_ocr(self, image_path):
        from manga_ocr import MangaOCR # Lazy Import
        if not self.model:
            self.model = MangaOCR()
        return self.model(image_path)

    def _ocr_tesseract(self, image_path):
        import pytesseract # Lazy Import
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return pytesseract.image_to_string(gray)
