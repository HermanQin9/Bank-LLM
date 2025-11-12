"""LLM engine module initialization."""

# Legacy client (for backward compatibility)
from src.llm_engine.gemini_client import GeminiClient

# New universal clients
from src.llm_engine.universal_client import UniversalLLMClient
from src.llm_engine.gemini_client_v2 import GeminiClientV2
from src.llm_engine.huggingface_client import HuggingFaceClient
from src.llm_engine.groq_client import GroqClient
from src.llm_engine.openrouter_client import OpenRouterClient
from src.llm_engine.base_llm_client import BaseLLMClient

# Utilities
from src.llm_engine.prompt_templates import PromptTemplates
from src.llm_engine.prompt_optimizer import PromptOptimizer

__all__ = [
    # Primary interface
    "UniversalLLMClient",
    
    # Individual clients
    "GeminiClientV2",
    "HuggingFaceClient",
    "GroqClient",
    "OpenRouterClient",
    "BaseLLMClient",
    
    # Legacy
    "GeminiClient",
    
    # Utilities
    "PromptTemplates",
    "PromptOptimizer",
]
