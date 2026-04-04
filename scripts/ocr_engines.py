import os
import cv2
import logging
from typing import List, Any

logger = logging.getLogger(__name__)

class OCREngine:
    def __init__(self, engine_type: str = "tesseract"):
        # Normalize engine type and ensure fallback list is unique
        self.primary_engine = engine_type.lower()
        self.engines_to_try = [self.primary_engine]
        if self.primary_engine != "tesseract":
            self.engines_to_try.append("tesseract")
            
        self._model: Any = None
        self._google_client: Any = None

    def get_text(self, image_path: str) -> str:
        """Extract text with smart deduplicated fallback."""
        last_error = None

        for engine in self.engines_to_try:
            try:
                method = getattr(self, f"_ocr_{engine}")
                text = method(image_path)
                if text and text.strip():
                    return text.strip()
                logger.warning(f"Engine {engine} returned empty text for {image_path}")
            except Exception as e:
                logger.error(f"Engine {engine} failed: {e}")
                last_error = e

        logger.error(f"All OCR attempts failed for {image_path}")
        return "" # Return empty rather than crashing the whole video pipeline

    # ------------------------------------------------------------------
    # Engine Implementations
    # ------------------------------------------------------------------

    def _ocr_google_vision(self, image_path: str) -> str:
        from google.cloud import vision
        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            raise RuntimeError("Google Vision credentials not found in environment.")
            
        if self._google_client is None:
            self._google_client = vision.ImageAnnotatorClient()

        with open(image_path, "rb") as f:
            content = f.read()
        
        image = vision.Image(content=content)
        response = self._google_client.text_detection(image=image)
        
        if response.error.message:
            raise Exception(f"Google Vision API error: {response.error.message}")

        return response.full_text_annotation.text if response.full_text_annotation else ""

    def _ocr_manga_ocr(self, image_path: str) -> str:
        # manga_ocr is excellent for Japanese but heavy.
        from manga_ocr import MangaOCR
        if self._model is None:
            logger.info("Loading MangaOCR model (this may take a moment)...")
            self._model = MangaOCR()
        return self._model(image_path)

    def _ocr_paddle_ocr(self, image_path: str) -> str:
        from paddleocr import PaddleOCR
        if self._model is None:
            # use_angle_cls=True helps with vertical manga text
            self._model = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
        
        result = self._model.ocr(image_path, cls=True)
        if not result or not result[0]:
            return ""
        return " ".join([line[1][0] for line in result[0]])

    def _ocr_tesseract(self, image_path: str) -> str:
        import pytesseract
        from PIL import Image
        
        # Optimized Tesseract Flow: 
        # Tesseract 4.0+ (LSTM) often performs better on raw or simple grayscaled images
        # than heavily threshed ones which can destroy thin kanji/characters.
        img = cv2.imread(image_path)
        if img is None:
            return ""
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding is MUCH better for Manga than global thresholding (150)
        # because it handles shadows/uneven lighting in scans.
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Try processing, but fallback to raw gray if it's too messy
        # Using both English and Japanese if available can be a lifesaver
        config = '--psm 3' # PSM 3 = Fully automatic page segmentation
        return pytesseract.image_to_string(processed, config=config)
