"""Base LLM client interface for universal implementation."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize base LLM client.
        
        Args:
            api_key: API key for the LLM service
            model_name: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def generate_with_metadata(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate text with metadata.
        
        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            Dictionary containing response text and metadata
        """
        pass
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Multi-turn conversation (default implementation uses generate).
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Response text
        """
        # Default implementation: convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        return self.generate(prompt)
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert message list to prompt string.
        
        Args:
            messages: List of messages
            
        Returns:
            Formatted prompt
        """
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n".join(prompt_parts)
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text (default implementation).
        
        Args:
            text: Input text
            
        Returns:
            Approximate token count
        """
        # Simple approximation: 1 token ≈ 4 characters
        return len(text) // 4
