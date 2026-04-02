import os
import cv2
import logging
from PIL import Image

logger = logging.getLogger(__name__)

class OCREngine:
    def __init__(self, engine_type="tesseract"):
        self.engine_type = engine_type
        self.model = None

    def get_text(self, image_path):
        try:
            if self.engine_type == "google_vision":
                return self._ocr_google_vision(image_path)
            elif self.engine_type == "manga_ocr":
                return self._ocr_manga_ocr(image_path)
            # [span_3](start_span)Default to Tesseract as safety net[span_3](end_span)
            return self._ocr_tesseract(image_path)
        except Exception as e:
            logger.error(f"Primary engine failed, falling back to Tesseract: {e}")
            return self._ocr_tesseract(image_path)

    def _ocr_manga_ocr(self, image_path):
        from manga_ocr import MangaOCR # Lazy Import
        if not self.model:
            self.model = MangaOCR()
        return self.model(image_path)

    def _ocr_tesseract(self, image_path):
        import pytesseract # Lazy Import
        img = cv2.imread(image_path)
        return pytesseract.image_to_string(img)
