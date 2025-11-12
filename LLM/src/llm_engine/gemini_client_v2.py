"""Google Gemini LLM client implementation using REST API."""

import requests
import json
from typing import Optional, Dict, List, Any
from src.utils import Config, logger
from src.llm_engine.base_llm_client import BaseLLMClient


class GeminiClientV2(BaseLLMClient):
    """Client for interacting with Google Gemini API using REST API."""
    
    # Gemini REST API endpoint
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    
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
            model_name: Model name (uses Config.GEMINI_MODEL if not provided)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key or Config.GOOGLE_API_KEY
        # Remove 'models/' prefix if present
        raw_model = model_name or Config.GEMINI_MODEL
        self.model_name = raw_model.replace('models/', '') if raw_model.startswith('models/') else raw_model
        self.temperature = temperature or Config.GEMINI_TEMPERATURE
        self.max_tokens = max_tokens or Config.GEMINI_MAX_TOKENS
        self.logger = logger
        
        self.logger.info(f"Initialized Gemini client with model: {self.model_name}")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text using Gemini REST API.
        
        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            Generated text
        """
        try:
            url = f"{self.BASE_URL}/models/{self.model_name}:generateContent?key={self.api_key}"
            
            headers = {
                'Content-Type': 'application/json',
            }
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": temperature or self.temperature,
                    "maxOutputTokens": max_tokens or self.max_tokens,
                }
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                self.logger.debug(f"Gemini response data keys: {data.keys()}")
                
                if 'candidates' in data and len(data['candidates']) > 0:
                    candidate = data['candidates'][0]
                    self.logger.debug(f"Candidate keys: {candidate.keys()}")
                    
                    # Try to extract text even if finish reason is MAX_TOKENS
                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts = candidate['content']['parts']
                        if len(parts) > 0 and 'text' in parts[0]:
                            text_result = parts[0]['text']
                            self.logger.debug(f"Extracted text length: {len(text_result)}")
                            
                            # Log finish reason for debugging
                            finish_reason = candidate.get('finishReason', 'STOP')
                            if finish_reason != 'STOP':
                                self.logger.debug(f"Gemini finish reason: {finish_reason}")
                            
                            return text_result
                    
                    # Check for finishReason if no text found
                    finish_reason = candidate.get('finishReason', 'UNKNOWN')
                    self.logger.warning(f"Gemini finish reason: {finish_reason}, no text extracted. Candidate: {candidate}")
                else:
                    self.logger.warning(f"No candidates in response: {data}")
                
                return ""
            else:
                error_msg = f"Gemini API error {response.status_code}: {response.text[:200]}"
                self.logger.error(error_msg)
                return ""
                
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
        Generate text with metadata using REST API.
        
        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            Dictionary containing response text and metadata
        """
        try:
            url = f"{self.BASE_URL}/models/{self.model_name}:generateContent?key={self.api_key}"
            
            headers = {
                'Content-Type': 'application/json',
            }
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": temperature or self.temperature,
                    "maxOutputTokens": max_tokens or self.max_tokens,
                }
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                result_text = ""
                
                if 'candidates' in data and len(data['candidates']) > 0:
                    candidate = data['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts = candidate['content']['parts']
                        if len(parts) > 0 and 'text' in parts[0]:
                            result_text = parts[0]['text']
                
                # Extract usage metadata
                usage = data.get('usageMetadata', {})
                
                return {
                    'text': result_text,
                    'model': self.model_name,
                    'provider': 'gemini',
                    'prompt_tokens': usage.get('promptTokenCount', 0),
                    'response_tokens': usage.get('candidatesTokenCount', 0),
                    'total_tokens': usage.get('totalTokenCount', 0),
                    'safety_ratings': data.get('candidates', [{}])[0].get('safetyRatings', []) if data.get('candidates') else None,
                }
            else:
                error_msg = f"Gemini API error {response.status_code}: {response.text[:200]}"
                self.logger.error(error_msg)
                return {
                    'text': '',
                    'model': self.model_name,
                    'provider': 'gemini',
                    'error': error_msg
                }
                
        except Exception as e:
            self.logger.error(f"Gemini generation with metadata failed: {e}")
            return {
                'text': '',
                'model': self.model_name,
                'provider': 'gemini',
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
            # Convert messages to context and prompt
            context = ""
            prompt = ""
            
            for msg in messages[:-1]:
                if msg['role'] == 'system':
                    context += msg['content'] + "\n"
                elif msg['role'] == 'user':
                    context += "User: " + msg['content'] + "\n"
                elif msg['role'] == 'assistant':
                    context += "Assistant: " + msg['content'] + "\n"
            
            if messages:
                prompt = messages[-1]['content']
            
            full_prompt = f"{context}\nUser: {prompt}\nAssistant:"
            
            return self.generate(full_prompt)
            
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
            # Use Gemini's token counting API
            url = f"{self.BASE_URL}/models/{self.model_name}:countTokens?key={self.api_key}"
            
            headers = {
                'Content-Type': 'application/json',
            }
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": text
                    }]
                }]
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('totalTokens', len(text.split()))
        except Exception as e:
            self.logger.warning(f"Token counting failed, using approximation: {e}")
        
        # Fallback to word-based approximation (roughly 1.3 words per token)
        return int(len(text.split()) * 1.3)
