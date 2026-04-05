import os
import cv2
import logging
import importlib.util
import re
from typing import List, Any
from tempfile import NamedTemporaryFile

# Global environment flags for Paddle optimization
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_allocator_strategy"] = "naive_best_fit"

logger = logging.getLogger(__name__)

class OCREngine:
    def __init__(self, engine_type: str = "manga_ocr"):
        # Prioritize manga_ocr for better handling of stylized fonts
        self.primary_engine = engine_type.lower()
        self.engines_to_try = [self.primary_engine]
        
        if self.primary_engine != "tesseract":
            self.engines_to_try.append("tesseract")
            
        self._model = None
        self._google_client = None

    def _preprocess_for_ocr(self, image_path: str) -> str:
        """
        Cleans the image by removing noise and forced binarization.
        This prevents the OCR from 'reading' background art textures.
        """
        img = cv2.imread(image_path)
        if img is None: 
            return image_path

        # 1. Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Denoise (Removes screentones/dots common in manga)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # 3. Otsu Thresholding (Converts to pure Black and White)
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Save to temp file for engine consumption
        temp_file = NamedTemporaryFile(delete=False, suffix=".png").name
        cv2.imwrite(temp_file, thresh)
        return temp_file

    def get_text(self, image_path: str) -> str:
        ready_image = self._preprocess_for_ocr(image_path)
        
        for engine in self.engines_to_try:
            try:
                method = getattr(self, f"_ocr_{engine}")
                text = method(ready_image)
                
                if text:
                    # CLEANING STEP: Remove junk symbols (backslashes, underscores, brackets)
                    # Keep only letters, numbers, and basic punctuation
                    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text)
                    # Standardize whitespace
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    if len(text) > 1:
                        if ready_image != image_path and os.path.exists(ready_image):
                            os.remove(ready_image)
                        return text
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
        return pytesseract.image_to_string(img, config='--psm 3')
