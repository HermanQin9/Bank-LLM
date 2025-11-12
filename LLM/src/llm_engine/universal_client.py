"""Universal LLM client with multi-provider support and auto-fallback."""

from typing import Optional, Dict, List, Any
from src.utils import Config, logger
from src.llm_engine.base_llm_client import BaseLLMClient


class UniversalLLMClient:
    """
    Universal LLM client that supports multiple providers with auto-fallback.
    
    Supported providers:
    - gemini: Google Gemini Pro
    - huggingface: Hugging Face Inference API
    - groq: Groq (fast inference, 14K free requests/day)
    - openrouter: OpenRouter (access to 100+ models)
    - auto: Automatically select available provider
    """
    
    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_fallback: bool = True
    ):
        """
        Initialize Universal LLM client.
        
        Args:
            provider: LLM provider ('gemini', 'huggingface', or 'auto')
            api_key: API key for the provider
            model_name: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            enable_fallback: Enable automatic fallback to other providers
        """
        self.provider = provider or Config.DEFAULT_LLM_PROVIDER
        self.enable_fallback = enable_fallback
        self.logger = logger
        
        # Initialize primary client
        self.client = self._initialize_client(
            self.provider, api_key, model_name, temperature, max_tokens
        )
        
        # Initialize fallback clients
        self.fallback_clients = []
        if self.enable_fallback:
            self.fallback_clients = self._initialize_fallback_clients(self.provider)
        
        self.logger.info(
            f"Initialized Universal LLM Client with provider: {self.provider}, "
            f"fallback: {len(self.fallback_clients)} providers"
        )
    
    def _initialize_client(
        self,
        provider: str,
        api_key: Optional[str],
        model_name: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int]
    ) -> BaseLLMClient:
        """Initialize a client for the specified provider."""
        if provider == 'auto':
            # Auto-select based on available API keys
            if Config.GROQ_API_KEY:
                provider = 'groq'
            elif Config.OPENROUTER_API_KEY:
                provider = 'openrouter'
            elif Config.GOOGLE_API_KEY:
                provider = 'gemini'
            elif Config.HUGGINGFACE_API_KEY:
                provider = 'huggingface'
            else:
                raise ValueError("No LLM provider API key configured")
        
        if provider == 'gemini':
            from src.llm_engine.gemini_client_v2 import GeminiClientV2
            return GeminiClientV2(api_key, model_name, temperature, max_tokens)
        
        elif provider == 'huggingface':
            from src.llm_engine.huggingface_client import HuggingFaceClient
            return HuggingFaceClient(api_key, model_name, temperature, max_tokens)
        
        elif provider == 'groq':
            from src.llm_engine.groq_client import GroqClient
            return GroqClient(api_key, model_name, temperature, max_tokens)
        
        elif provider == 'openrouter':
            from src.llm_engine.openrouter_client import OpenRouterClient
            return OpenRouterClient(api_key, model_name, temperature, max_tokens)
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _initialize_fallback_clients(self, primary_provider: str) -> List[BaseLLMClient]:
        """Initialize fallback clients for other available providers."""
        fallback_clients = []
        
        # Priority order: Groq > OpenRouter > Gemini > Hugging Face
        if primary_provider != 'groq' and Config.GROQ_API_KEY:
            try:
                from src.llm_engine.groq_client import GroqClient
                fallback_clients.append(GroqClient())
                self.logger.info("Added Groq as fallback provider")
            except Exception as e:
                self.logger.warning(f"Could not initialize Groq fallback: {e}")
        
        if primary_provider != 'openrouter' and Config.OPENROUTER_API_KEY:
            try:
                from src.llm_engine.openrouter_client import OpenRouterClient
                fallback_clients.append(OpenRouterClient())
                self.logger.info("Added OpenRouter as fallback provider")
            except Exception as e:
                self.logger.warning(f"Could not initialize OpenRouter fallback: {e}")
        
        if primary_provider != 'gemini' and Config.GOOGLE_API_KEY:
            try:
                from src.llm_engine.gemini_client_v2 import GeminiClientV2
                fallback_clients.append(GeminiClientV2())
                self.logger.info("Added Gemini as fallback provider")
            except Exception as e:
                self.logger.warning(f"Could not initialize Gemini fallback: {e}")
        
        if primary_provider != 'huggingface' and Config.HUGGINGFACE_API_KEY:
            try:
                from src.llm_engine.huggingface_client import HuggingFaceClient
                fallback_clients.append(HuggingFaceClient())
                self.logger.info("Added Hugging Face as fallback provider")
            except Exception as e:
                self.logger.warning(f"Could not initialize Hugging Face fallback: {e}")
        
        return fallback_clients
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text using the configured provider with auto-fallback.
        
        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            Generated text
        """
        # Try primary client
        try:
            response = self.client.generate(prompt, temperature, max_tokens)
            if response:
                return response
        except Exception as e:
            self.logger.warning(f"Primary provider ({self.provider}) failed: {e}")
        
        # Try fallback clients
        if self.enable_fallback:
            for i, fallback_client in enumerate(self.fallback_clients):
                try:
                    self.logger.info(f"Trying fallback provider {i+1}/{len(self.fallback_clients)}")
                    response = fallback_client.generate(prompt, temperature, max_tokens)
                    if response:
                        return response
                except Exception as e:
                    self.logger.warning(f"Fallback provider {i+1} failed: {e}")
        
        self.logger.error("All providers failed to generate response")
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
        # Try primary client
        try:
            result = self.client.generate_with_metadata(prompt, temperature, max_tokens)
            if result.get('text'):
                return result
        except Exception as e:
            self.logger.warning(f"Primary provider ({self.provider}) failed: {e}")
        
        # Try fallback clients
        if self.enable_fallback:
            for i, fallback_client in enumerate(self.fallback_clients):
                try:
                    self.logger.info(f"Trying fallback provider {i+1}/{len(self.fallback_clients)}")
                    result = fallback_client.generate_with_metadata(prompt, temperature, max_tokens)
                    if result.get('text'):
                        result['used_fallback'] = True
                        result['fallback_index'] = i
                        return result
                except Exception as e:
                    self.logger.warning(f"Fallback provider {i+1} failed: {e}")
        
        return {
            'text': '',
            'error': 'All providers failed',
            'provider': self.provider
        }
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Multi-turn conversation.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Response text
        """
        # Try primary client
        try:
            response = self.client.chat(messages)
            if response:
                return response
        except Exception as e:
            self.logger.warning(f"Primary provider ({self.provider}) chat failed: {e}")
        
        # Try fallback clients
        if self.enable_fallback:
            for i, fallback_client in enumerate(self.fallback_clients):
                try:
                    self.logger.info(f"Trying fallback provider {i+1} for chat")
                    response = fallback_client.chat(messages)
                    if response:
                        return response
                except Exception as e:
                    self.logger.warning(f"Fallback provider {i+1} chat failed: {e}")
        
        return ""
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Input text
            
        Returns:
            Approximate token count
        """
        return self.client.count_tokens(text)
    
    @property
    def model_name(self) -> str:
        """Get the model name of the current provider."""
        return self.client.model_name
    
    @property
    def temperature(self) -> float:
        """Get the temperature setting."""
        return self.client.temperature
    
    @property
    def max_tokens(self) -> int:
        """Get the max tokens setting."""
        return self.client.max_tokens
