"""Groq API LLM client implementation (Fast & Free)."""

import requests
from typing import Optional, Dict, List, Any
from src.utils import Config
from src.llm_engine.base_llm_client import BaseLLMClient


class GroqClient(BaseLLMClient):
    """
    Client for Groq API (Ultra-fast inference).
    
    Free tier: 14,400 requests/day
    Models: Llama 3, Mixtral, Gemma
    Speed: Fastest LLM API available
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize Groq client.
        
        Args:
            api_key: Groq API key (get from https://console.groq.com)
            model_name: Model name (e.g., llama3-8b-8192, mixtral-8x7b-32768)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key or Config.GROQ_API_KEY
        self.model_name = model_name or Config.GROQ_MODEL
        self.temperature = temperature or Config.GROQ_TEMPERATURE
        self.max_tokens = max_tokens or Config.GROQ_MAX_TOKENS
        
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        print(f"Initialized Groq client with model: {self.model_name}")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text using Groq API."""
        try:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                print(f"Groq API error {response.status_code}: {response.text}")
                return ""
                
        except Exception as e:
            print(f"Groq generation failed: {e}")
            return ""
    
    def generate_with_metadata(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate text with metadata."""
        try:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'text': result['choices'][0]['message']['content'].strip(),
                    'model': self.model_name,
                    'provider': 'groq',
                    'prompt_tokens': result['usage']['prompt_tokens'],
                    'response_tokens': result['usage']['completion_tokens'],
                    'total_tokens': result['usage']['total_tokens'],
                }
            else:
                return {
                    'text': '',
                    'model': self.model_name,
                    'provider': 'groq',
                    'error': f"Status {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                'text': '',
                'model': self.model_name,
                'provider': 'groq',
                'error': str(e)
            }
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Multi-turn conversation."""
        try:
            # Convert to Groq format
            groq_messages = []
            for msg in messages:
                groq_messages.append({
                    "role": msg.get('role', 'user'),
                    "content": msg.get('content', '')
                })
            
            payload = {
                "model": self.model_name,
                "messages": groq_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            return ""
            
        except Exception as e:
            print(f"Groq chat failed: {e}")
            return ""
