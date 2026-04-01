import os
import cv2
import torch
import numpy as np
from PIL import Image
import pytesseract
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCREngine:
    """OCR Engine with comprehensive failover and fallback mechanisms."""
    
    def __init__(self, engine_type="tesseract"):
        self.engine_type = engine_type
        self.models = {}  # Store loaded models
        self.initialized = False
        logger.info(f"Initialized OCR Engine: {self.engine_type}")

    def get_text(self, image_path):
        """Main entry point for text extraction with failover protection."""
        # Validate image exists
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return ""
        
        # Validate image is readable
        try:
            test_img = Image.open(image_path)
            test_img.verify()
        except Exception as e:
            logger.error(f"Image corrupted or unreadable: {image_path}, error: {e}")
            return ""
        
        # Try the selected engine with fallbacks
        engines_to_try = [self.engine_type, "tesseract"]  # Always fallback to tesseract
        
        for engine in engines_to_try:
            try:
                if engine == "google_vision":
                    result = self._ocr_google_vision(image_path)
                elif engine == "manga_ocr":
                    result = self._ocr_manga_ocr(image_path)
                elif engine == "paddle_ocr":
                    result = self._ocr_paddle(image_path)
                elif engine == "comic_text_detector":
                    result = self._ocr_comic_detector(image_path)
                else:
                    result = self._ocr_tesseract(image_path)
                
                if result and result.strip():
                    logger.info(f"OCR successful using {engine}, extracted {len(result)} chars")
                    return result.strip()
                else:
                    logger.warning(f"OCR engine {engine} returned empty result")
                    
            except Exception as e:
                logger.error(f"OCR engine {engine} failed: {e}")
                continue
        
        # Ultimate fallback - return empty string
        logger.error(f"All OCR engines failed for {image_path}")
        return ""

    def _ocr_google_vision(self, image_path):
        """Google Cloud Vision OCR."""
        try:
            from google.cloud import vision
            import io
        except ImportError:
            logger.error("Google Cloud Vision not installed")
            raise ImportError("google-cloud-vision not installed")
        
        # Try to load credentials from environment
        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path or not os.path.exists(credentials_path):
            raise Exception("Google credentials not configured")
        
        try:
            client = vision.ImageAnnotatorClient()
            with io.open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            response = client.text_detection(image=image)
            
            if response.error.message:
                raise Exception(f"Google Vision Error: {response.error.message}")
            
            if response.full_text_annotation and response.full_text_annotation.text:
                return response.full_text_annotation.text
            return ""
            
        except Exception as e:
            logger.error(f"Google Vision processing error: {e}")
            raise

    def _ocr_manga_ocr(self, image_path):
        """Manga-OCR specialized for Japanese manga text."""
        model_key = "manga_ocr"
        
        if model_key not in self.models:
            try:
                from manga_ocr import MangaOCR
                self.models[model_key] = MangaOCR()
                logger.info("Manga-OCR model loaded successfully")
            except ImportError as e:
                logger.error(f"Manga-OCR import failed: {e}")
                raise
            except Exception as e:
                logger.error(f"Manga-OCR initialization error: {e}")
                raise
        
        try:
            # MangaOCR expects image path or PIL Image
            result = self.models[model_key](image_path)
            return result.strip() if result else ""
        except Exception as e:
            logger.error(f"Manga-OCR processing error: {e}")
            raise

    def _ocr_paddle(self, image_path):
        """PaddleOCR with memory optimization."""
        model_key = "paddle_ocr"
        
        if model_key not in self.models:
            try:
                from paddleocr import PaddleOCR
                self.models[model_key] = PaddleOCR(
                    use_angle_cls=True, 
                    lang='en', 
                    use_gpu=False, 
                    show_log=False,
                    enable_mkldnn=True,
                    use_tensorrt=False,
                    det_db_thresh=0.3,
                    det_db_box_thresh=0.5
                )
                logger.info("PaddleOCR model loaded successfully")
            except ImportError as e:
                logger.error(f"PaddleOCR import failed: {e}")
                raise
            except Exception as e:
                logger.error(f"PaddleOCR initialization error: {e}")
                raise
        
        try:
            result = self.models[model_key].ocr(image_path, cls=True)
            
            if not result:
                return ""
            
            full_text = []
            for line_results in result:
                if line_results:
                    for line in line_results:
                        if line and len(line) >= 2 and line[1]:
                            full_text.append(line[1][0])
            
            return " ".join(full_text).strip()
            
        except Exception as e:
            logger.error(f"PaddleOCR processing error: {e}")
            raise

    def _ocr_comic_detector(self, image_path):
        """Specialized comic text detection with bubble detection."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            bubble_texts = []
            img_height, img_width = img.shape[:2]
            
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                img_area = img_width * img_height
                
                # Filter for speech bubble candidates
                if (w > 30 and h > 20 and 
                    area > 200 and area < img_area * 0.3 and
                    w/h < 5 and h/w < 5):
                    
                    roi = img[y:y+h, x:x+w]
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    roi_gray = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    
                    text = pytesseract.image_to_string(roi_gray, config='--psm 7').strip()
                    if text and len(text) > 1:
                        bubble_texts.append(text)
            
            if bubble_texts:
                return " ".join(bubble_texts)
            else:
                logger.info("No speech bubbles detected, falling back to full image OCR")
                return self._ocr_tesseract(image_path)
                
        except Exception as e:
            logger.error(f"Comic detector error: {e}")
            raise

    def _ocr_tesseract(self, image_path):
        """Ultimate fallback: Standard Tesseract OCR."""
        try:
            # Open image and convert to grayscale
            img = Image.open(image_path).convert('L')
            
            # Apply preprocessing for better results
            img_array = np.array(img)
            img_array = cv2.equalizeHist(img_array)
            img = Image.fromarray(img_array)
            
            # Tesseract with optimized config
            config = '--psm 6 --oem 3'
            text = pytesseract.image_to_string(img, config=config)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
            return ""

    def cleanup(self):
        """Clean up loaded models to free memory."""
        self.models.clear()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
