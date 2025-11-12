"""Tests for LLM engine modules."""

import pytest
from src.llm_engine import PromptTemplates


class TestPromptTemplates:
    """Test PromptTemplates class."""
    
    def test_document_classification(self):
        """Test classification prompt generation."""
        prompt = PromptTemplates.document_classification(
            "Sample document text",
            ["Category A", "Category B"]
        )
        assert "Category A" in prompt
        assert "Category B" in prompt
        assert "Sample document text" in prompt
    
    def test_information_extraction(self):
        """Test extraction prompt generation."""
        prompt = PromptTemplates.information_extraction(
            "Sample document",
            ["Name", "Date"]
        )
        assert "Name" in prompt
        assert "Date" in prompt
    
    def test_question_answering(self):
        """Test QA prompt generation."""
        prompt = PromptTemplates.question_answering(
            "Context text",
            "What is this?"
        )
        assert "Context text" in prompt
        assert "What is this?" in prompt


if __name__ == "__main__":
    pytest.main([__file__])
