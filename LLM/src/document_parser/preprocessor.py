"""Text preprocessing utilities."""

import re
from typing import List, Dict
from src.utils import logger, clean_text, split_text_into_chunks


class TextPreprocessor:
    """Preprocess and normalize text from various sources."""
    
    def __init__(self):
        """Initialize text preprocessor."""
        self.logger = logger
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def remove_special_characters(self, text: str, keep_punctuation: bool = True) -> str:
        """Remove special characters from text."""
        if keep_punctuation:
            # Keep letters, numbers, and basic punctuation
            text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\-\(\)\[\]\"\']+', '', text)
        else:
            # Keep only letters and numbers
            text = re.sub(r'[^a-zA-Z0-9\s]+', '', text)
        return text
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, any]]:
        """
        Split text into chunks for processing.
        
        Args:
            text: Input text
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        chunks = split_text_into_chunks(text, chunk_size, overlap)
        
        result = []
        for i, chunk in enumerate(chunks):
            result.append({
                'chunk_id': i,
                'text': chunk,
                'length': len(chunk),
                'start_pos': i * (chunk_size - overlap)
            })
        
        return result
    
    def preprocess(self, text: str, operations: List[str] = None) -> str:
        """
        Apply preprocessing operations to text.
        
        Args:
            text: Input text
            operations: List of operations to apply
                       ('normalize', 'clean_special', 'lowercase')
                       
        Returns:
            Preprocessed text
        """
        if operations is None:
            operations = ['normalize']
        
        result = text
        
        if 'normalize' in operations:
            result = self.normalize_whitespace(result)
        
        if 'clean_special' in operations:
            result = self.remove_special_characters(result)
        
        if 'lowercase' in operations:
            result = result.lower()
        
        return result
