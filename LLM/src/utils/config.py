"""Configuration management for the document intelligence system."""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration."""
    
    # LLM Provider Selection
    DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")
    
    # Google Gemini API
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-pro")
    GEMINI_MAX_TOKENS: int = int(os.getenv("GEMINI_MAX_TOKENS", "4096"))
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
    
    # Hugging Face API
    HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", "")
    HUGGINGFACE_MODEL: str = os.getenv("HUGGINGFACE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    HUGGINGFACE_MAX_TOKENS: int = int(os.getenv("HUGGINGFACE_MAX_TOKENS", "2048"))
    HUGGINGFACE_TEMPERATURE: float = float(os.getenv("HUGGINGFACE_TEMPERATURE", "0.7"))
    
    # Groq API (Fast & Free - 14K requests/day)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    GROQ_MAX_TOKENS: int = int(os.getenv("GROQ_MAX_TOKENS", "2048"))
    GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.7"))
    
    # OpenRouter API (Access 100+ models)
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "google/gemini-pro-1.5")
    OPENROUTER_MAX_TOKENS: int = int(os.getenv("OPENROUTER_MAX_TOKENS", "4096"))
    OPENROUTER_TEMPERATURE: float = float(os.getenv("OPENROUTER_TEMPERATURE", "0.7"))
    
    # Legacy Configuration (backward compatibility)
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gemini-pro")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "4096"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    
    # Application
    APP_ENV: str = os.getenv("APP_ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # Paths
    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
    RAW_DATA_DIR: str = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR: str = os.path.join(DATA_DIR, "processed")
    RESULTS_DIR: str = os.path.join(PROJECT_ROOT, "results")
    CONFIGS_DIR: str = os.path.join(PROJECT_ROOT, "configs")
    
    # Vector Database
    VECTOR_DB_PATH: str = os.path.join(DATA_DIR, "chroma_db")
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Document Processing
    MAX_FILE_SIZE_MB: int = 50
    SUPPORTED_FORMATS: list = [".pdf", ".txt", ".docx", ".png", ".jpg", ".jpeg"]
    
    # RAG System
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 5
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration."""
        # Check if at least one LLM provider is configured
        has_gemini = bool(cls.GOOGLE_API_KEY)
        has_huggingface = bool(cls.HUGGINGFACE_API_KEY)
        has_groq = bool(cls.GROQ_API_KEY)
        has_openrouter = bool(cls.OPENROUTER_API_KEY)
        
        if not (has_gemini or has_huggingface or has_groq or has_openrouter):
            print("Warning: No LLM provider API key configured.")
            print("Please set at least one of: GOOGLE_API_KEY, GROQ_API_KEY, or OPENROUTER_API_KEY")
        return True


# Validate on import
Config.validate()
