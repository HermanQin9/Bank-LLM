"""Generator for RAG system using Gemini."""

from typing import Dict, Optional, Any
from src.llm_engine import GeminiClient, PromptTemplates
from src.utils import logger


class Generator:
    """Generator for producing answers using retrieved context."""
    
    def __init__(
        self,
        llm_client: Optional[GeminiClient] = None
    ):
        """
        Initialize generator.
        
        Args:
            llm_client: GeminiClient instance
        """
        self.llm_client = llm_client or GeminiClient()
        self.logger = logger
    
    def generate(
        self,
        query: str,
        context: str,
        include_sources: bool = True
    ) -> str:
        """
        Generate answer based on query and context.
        
        Args:
            query: User question
            context: Retrieved context
            include_sources: Whether to include source attribution
            
        Returns:
            Generated answer
        """
        prompt = PromptTemplates.question_answering(context, query)
        
        if include_sources:
            prompt += "\n\nPlease cite which document(s) support your answer."
        
        answer = self.llm_client.generate(prompt)
        
        return answer
    
    def generate_with_metadata(
        self,
        query: str,
        context: str,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Generate answer with metadata.
        
        Args:
            query: User question
            context: Retrieved context
            include_sources: Whether to include source attribution
            
        Returns:
            Dictionary with answer and metadata
        """
        prompt = PromptTemplates.question_answering(context, query)
        
        if include_sources:
            prompt += "\n\nPlease cite which document(s) support your answer."
        
        result = self.llm_client.generate_with_metadata(prompt)
        
        return {
            'answer': result['text'],
            'query': query,
            'context_length': len(context),
            'model': result.get('model'),
            'tokens': {
                'prompt': result.get('prompt_tokens', 0),
                'response': result.get('response_tokens', 0)
            }
        }
    
    def generate_summary(self, context: str, max_sentences: int = 3) -> str:
        """
        Generate summary of context.
        
        Args:
            context: Text to summarize
            max_sentences: Maximum sentences in summary
            
        Returns:
            Summary text
        """
        prompt = PromptTemplates.document_summary(context, max_sentences)
        summary = self.llm_client.generate(prompt)
        
        return summary
    
    def extract_information(
        self,
        context: str,
        fields: list
    ) -> Dict[str, str]:
        """
        Extract structured information from context.
        
        Args:
            context: Text to extract from
            fields: List of fields to extract
            
        Returns:
            Dictionary of extracted information
        """
        prompt = PromptTemplates.information_extraction(context, fields)
        response = self.llm_client.generate(prompt)
        
        # Parse response into dictionary
        extracted = {}
        for line in response.split('\n'):
            for field in fields:
                if line.startswith(f"{field}:"):
                    value = line.replace(f"{field}:", "").strip()
                    extracted[field] = value
                    break
        
        return extracted
