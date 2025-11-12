"""Tests for document parser modules."""

import pytest
from src.document_parser import PDFParser, TextPreprocessor


class TestTextPreprocessor:
    """Test TextPreprocessor class."""
    
    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        preprocessor = TextPreprocessor()
        text = "Hello    world  \n  test"
        result = preprocessor.normalize_whitespace(text)
        assert result == "Hello world test"
    
    def test_chunk_text(self):
        """Test text chunking."""
        preprocessor = TextPreprocessor()
        text = "a" * 1500
        chunks = preprocessor.chunk_text(text, chunk_size=500, overlap=100)
        
        assert len(chunks) > 1
        assert all(isinstance(chunk, dict) for chunk in chunks)
        assert all('text' in chunk for chunk in chunks)
    
    def test_preprocess(self):
        """Test preprocessing operations."""
        preprocessor = TextPreprocessor()
        text = "  Hello   World!  "
        result = preprocessor.preprocess(text, ['normalize'])
        assert result == "Hello World!"


if __name__ == "__main__":
    pytest.main([__file__])
