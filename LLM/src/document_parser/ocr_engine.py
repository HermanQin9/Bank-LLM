"""OCR engine for image and scanned document processing."""

from typing import Optional, Dict
from pathlib import Path
from PIL import Image
import pytesseract
from src.utils import logger, clean_text


class OCREngine:
    """OCR engine for extracting text from images."""
    
    def __init__(self, language: str = 'eng'):
        """
        Initialize OCR engine.
        
        Args:
            language: Tesseract language code (default: 'eng' for English)
        """
        self.language = language
        self.logger = logger
        
    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from image using Tesseract OCR.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text
        """
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang=self.language)
            return clean_text(text)
        except Exception as e:
            self.logger.error(f"OCR extraction failed for {image_path}: {e}")
            return ""
    
    def extract_with_confidence(self, image_path: str) -> Dict[str, any]:
        """
        Extract text with confidence scores.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with text and confidence information
        """
        try:
            image = Image.open(image_path)
            data = pytesseract.image_to_data(image, lang=self.language, output_type=pytesseract.Output.DICT)
            
            # Filter out low confidence text
            text_segments = []
            confidences = []
            
            for i, conf in enumerate(data['conf']):
                if int(conf) > 0:  # Valid confidence score
                    text = data['text'][i].strip()
                    if text:
                        text_segments.append(text)
                        confidences.append(int(conf))
            
            full_text = " ".join(text_segments)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': clean_text(full_text),
                'confidence': avg_confidence,
                'num_segments': len(text_segments)
            }
        except Exception as e:
            self.logger.error(f"OCR with confidence failed for {image_path}: {e}")
            return {'text': '', 'confidence': 0, 'num_segments': 0}
    
    def process(self, filepath: str) -> Dict[str, any]:
        """
        Process image file and extract text.
        
        Args:
            filepath: Path to image file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        self.logger.info(f"Processing image with OCR: {filepath}")
        
        if not Path(filepath).exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        result = self.extract_with_confidence(filepath)
        result['filepath'] = filepath
        result['filename'] = Path(filepath).name
        
        self.logger.info(f"Extracted {len(result['text'])} characters with {result['confidence']:.2f}% confidence")
        
        return result
