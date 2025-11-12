"""Hugging Face API LLM client implementation."""

import requests
from typing import Optional, Dict, List, Any
from src.utils import Config, logger
from src.llm_engine.base_llm_client import BaseLLMClient


class HuggingFaceClient(BaseLLMClient):
    """Client for interacting with Hugging Face Inference API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize Hugging Face client.
        
        Args:
            api_key: Hugging Face API key
            model_name: Model name on Hugging Face Hub
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key or Config.HUGGINGFACE_API_KEY
        self.model_name = model_name or Config.HUGGINGFACE_MODEL
        self.temperature = temperature or Config.HUGGINGFACE_TEMPERATURE
        self.max_tokens = max_tokens or Config.HUGGINGFACE_MAX_TOKENS
        self.logger = logger
        
        # API endpoint (use free inference API)
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        self.logger.info(f"Initialized Hugging Face client with model: {self.model_name}")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text using Hugging Face API.
        
        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            Generated text
        """
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": temperature or self.temperature,
                    "max_new_tokens": max_tokens or self.max_tokens,
                    "return_full_text": False,
                    "do_sample": True,
                }
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Handle different response formats
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict) and 'generated_text' in result[0]:
                        return result[0]['generated_text'].strip()
                    elif isinstance(result[0], str):
                        return result[0].strip()
                elif isinstance(result, dict) and 'generated_text' in result:
                    return result['generated_text'].strip()
                
                return str(result)
            else:
                error_msg = f"API returned status {response.status_code}: {response.text}"
                self.logger.error(f"Hugging Face API error: {error_msg}")
                return ""
                
        except Exception as e:
            self.logger.error(f"Hugging Face generation failed: {e}")
            return ""
    
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
        try:
            text = self.generate(prompt, temperature, max_tokens)
            
            return {
                'text': text,
                'model': self.model_name,
                'provider': 'huggingface',
                'prompt_tokens': len(prompt.split()),
                'response_tokens': len(text.split()) if text else 0,
            }
        except Exception as e:
            self.logger.error(f"Hugging Face generation with metadata failed: {e}")
            return {
                'text': '',
                'model': self.model_name,
                'provider': 'huggingface',
                'error': str(e)
            }
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Multi-turn conversation.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Response text
        """
        # Convert messages to prompt format
        prompt = self._format_chat_prompt(messages)
        return self.generate(prompt)
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Format chat messages for the model.
        
        Args:
            messages: List of messages
            
        Returns:
            Formatted prompt
        """
        # Use Mistral/Llama chat format
        formatted_parts = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                formatted_parts.append(f"<s>[INST] {content} [/INST]</s>")
            elif role == 'user':
                formatted_parts.append(f"<s>[INST] {content} [/INST]")
            elif role == 'assistant':
                formatted_parts.append(f"{content}</s>")
        
        return "\n".join(formatted_parts)
