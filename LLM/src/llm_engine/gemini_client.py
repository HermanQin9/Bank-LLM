"""Gemini LLM client for document understanding."""

import google.generativeai as genai
from typing import Optional, Dict, List, Any
from src.utils import Config, logger


class GeminiClient:
    """Client for interacting with Google Gemini API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google API key (uses Config.GOOGLE_API_KEY if not provided)
            model_name: Model name (uses Config.DEFAULT_MODEL if not provided)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key or Config.GOOGLE_API_KEY
        self.model_name = model_name or Config.DEFAULT_MODEL
        self.temperature = temperature or Config.TEMPERATURE
        self.max_tokens = max_tokens or Config.MAX_TOKENS
        self.logger = logger
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize model (use older API if GenerativeModel not available)
        if hasattr(genai, 'GenerativeModel'):
            self.model = genai.GenerativeModel(self.model_name)
            self.use_new_api = True
        else:
            self.model = None
            self.use_new_api = False
        
        self.logger.info(f"Initialized Gemini client with model: {self.model_name}")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text using Gemini.
        
        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            Generated text
        """
        try:
            if self.use_new_api and self.model:
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature or self.temperature,
                    max_output_tokens=max_tokens or self.max_tokens,
                )
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                return response.text
            else:
                # Use older API
                response = genai.generate_text(
                    model=f'models/{self.model_name}',
                    prompt=prompt,
                    temperature=temperature or self.temperature,
                    max_output_tokens=max_tokens or self.max_tokens,
                )
                return response.result if response else ""
                
        except Exception as e:
            self.logger.error(f"Gemini generation failed: {e}")
            return ""
    
    def generate_with_metadata(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate text with metadata (token usage, etc.).
        
        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            Dictionary containing response text and metadata
        """
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature or self.temperature,
                max_output_tokens=max_tokens or self.max_tokens,
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return {
                'text': response.text,
                'model': self.model_name,
                'finish_reason': response.candidates[0].finish_reason if response.candidates else None,
                'safety_ratings': response.candidates[0].safety_ratings if response.candidates else None,
                'prompt_tokens': len(prompt.split()),  # Approximate
                'response_tokens': len(response.text.split()) if response.text else 0
            }
        except Exception as e:
            self.logger.error(f"Gemini generation with metadata failed: {e}")
            return {
                'text': '',
                'model': self.model_name,
                'error': str(e)
            }
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Multi-turn conversation with Gemini.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Response text
        """
        try:
            # Convert messages to Gemini format
            chat = self.model.start_chat(history=[])
            
            for msg in messages[:-1]:  # Add history
                if msg['role'] == 'user':
                    chat.send_message(msg['content'])
            
            # Send last message and get response
            if messages:
                response = chat.send_message(messages[-1]['content'])
                return response.text
            
            return ""
        except Exception as e:
            self.logger.error(f"Gemini chat failed: {e}")
            return ""
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Input text
            
        Returns:
            Approximate token count
        """
        try:
            result = self.model.count_tokens(text)
            return result.total_tokens
        except Exception as e:
            self.logger.warning(f"Token counting failed, using approximation: {e}")
            # Fallback to word-based approximation
            return len(text.split())
