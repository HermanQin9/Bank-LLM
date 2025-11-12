"""OpenRouter API client - Access to 100+ LLMs with single API key."""

import requests
from typing import Optional, Dict, List, Any
from src.utils import Config
from src.llm_engine.base_llm_client import BaseLLMClient


class OpenRouterClient(BaseLLMClient):
    """
    Client for OpenRouter API.
    
    OpenRouter provides access to 100+ models through one API:
    - Free models: Google Gemini, Meta Llama, Mistral
    - Paid models: GPT-4, Claude, etc.
    - Pay-as-you-go pricing
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key (get from https://openrouter.ai/keys)
            model_name: Model ID (e.g., google/gemini-pro-1.5, meta-llama/llama-3-8b-instruct)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key or Config.OPENROUTER_API_KEY
        self.model_name = model_name or Config.OPENROUTER_MODEL
        self.temperature = temperature or Config.OPENROUTER_TEMPERATURE
        self.max_tokens = max_tokens or Config.OPENROUTER_MAX_TOKENS
        
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/HermanQin9/LLM",  # Optional but recommended
            "X-Title": "LLM Document Intelligence System",  # Optional: App name
        }
        
        print(f"Initialized OpenRouter client with model: {self.model_name}")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text using OpenRouter API."""
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
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                print(f"OpenRouter API error {response.status_code}: {response.text}")
                return ""
                
        except Exception as e:
            print(f"OpenRouter generation failed: {e}")
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
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                usage = result.get('usage', {})
                return {
                    'text': result['choices'][0]['message']['content'].strip(),
                    'model': self.model_name,
                    'provider': 'openrouter',
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'response_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0),
                }
            else:
                return {
                    'text': '',
                    'model': self.model_name,
                    'provider': 'openrouter',
                    'error': f"Status {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                'text': '',
                'model': self.model_name,
                'provider': 'openrouter',
                'error': str(e)
            }
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Multi-turn conversation."""
        try:
            # Convert to OpenRouter format
            openrouter_messages = []
            for msg in messages:
                openrouter_messages.append({
                    "role": msg.get('role', 'user'),
                    "content": msg.get('content', '')
                })
            
            payload = {
                "model": self.model_name,
                "messages": openrouter_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            return ""
            
        except Exception as e:
            print(f"OpenRouter chat failed: {e}")
            return ""
