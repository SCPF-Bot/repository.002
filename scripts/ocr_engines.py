import os
import cv2
import logging
import importlib.util
from typing import List, Any

# Suppress Paddle and Protobuf warnings in the console
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_allocator_strategy"] = "naive_best_fit"

logger = logging.getLogger(__name__)

class OCREngine:
    def __init__(self, engine_type: str = "tesseract"):
        self.primary_engine = engine_type.lower()
        self.engines_to_try = [self.primary_engine]
        
        # Always keep Tesseract as the final safety net
        if self.primary_engine != "tesseract":
            self.engines_to_try.append("tesseract")
            
        self._model = None
        self._google_client = None

    def get_text(self, image_path: str) -> str:
        """
        Loops through available engines. If the primary fails or 
        returns empty text, it tries the next one.
        """
        for engine in self.engines_to_try:
            try:
                method = getattr(self, f"_ocr_{engine}")
                text = method(image_path)
                if text and len(text.strip()) > 1:
                    return text.strip()
            except Exception as e:
                logger.error(f"Engine {engine} failed: {e}")
                continue
        return ""

    def _ocr_google_vision(self, image_path: str) -> str:
        # Check if google-cloud-vision is actually installed before importing
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
            logger.info("Loading MangaOCR model (this may take a moment)...")
            self._model = MangaOCR()
        return self._model(image_path)

    def _ocr_paddle_ocr(self, image_path: str) -> str:
        from paddleocr import PaddleOCR
        import logging as py_logging
        
        # Silence the internal paddleocr logger to keep GitHub Action logs clean
        py_logging.getLogger("ppocr").setLevel(py_logging.ERROR)

        if self._model is None:
            logger.info("Initializing PaddleOCR (CPU mode)...")
            # FIXED: Removed 'show_log' which caused the crash in v3.x
            self._model = PaddleOCR(
                use_angle_cls=True, 
                lang='en', 
                use_gpu=False,
                rec_batch_num=1
            )

        # Paddle OCR returns a list: [ [ [box, (text, score)], ... ] ]
        result = self._model.ocr(image_path, cls=True)
        
        if not result or result[0] is None:
            return ""
            
        # Extract text and join with spaces
        extracted_text = []
        for line in result[0]:
            text_val = line[1][0]
            extracted_text.append(text_val)
            
        return " ".join(extracted_text)

    def _ocr_tesseract(self, image_path: str) -> str:
        import pytesseract
        
        # Pre-processing for better Tesseract accuracy
        img = cv2.imread(image_path)
        if img is None:
            return ""
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use Adaptive Thresholding to handle varying lighting in manga scans
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # PSM 3: Fully automatic page segmentation, but no OSD.
        return pytesseract.image_to_string(processed, config='--psm 3')
